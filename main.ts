// gta6-twitch-ccu-forecast-01 — forecast Twitch concurrent viewers of Grand Theft
// Auto VI on its release day (2026-11-19 UTC), hours 2..24, five quantiles each.
//
// Model: at run time, learn the shape of a big console launch day from the
// `comparables` inside the case file itself. For every past launch, take the
// log ratio of each release-day hour's viewers to that launch's hour-1 print,
// then aggregate the ratios across launches with a similarity-weighted median
// and turn them into quantiles around this case's own hour-1 print:
//
//   q_tau(h) = p1 * exp(m_h + z_tau * s_h)
//
// where p1 is hour 1's print, m_h the (smoothed) weighted-median log ratio for
// hour h, z_tau the standard normal quantile at each scored level, and s_h the
// log spread for that hour.
//
// Similarity weights (all read from the case file, nothing hard-coded about any
// single game): rolling-midnight unlocks x3 (the shape this contest's launches
// share — viewers build as time zones unlock one by one), console-only launches
// x2, simultaneous global unlocks x0.25 (their hour 1 is pre-unlock noise, so
// their ratios explode and would swamp the median), early-access-contaminated
// launches x0.25 (their release day is not a clean launch), and a scale weight
// exp(-|ln(h1_peer / p1)|) so launches of a similar size count most.
//
// A second anchor steadies hour 1 when it is depressed: the pre-launch peak,
// the largest hourly count in the 24 h before the release day (for a rolling
// midnight launch New Zealand plays ~13 h before UTC midnight, so the evening
// before already shows real demand). When that peak exceeds hour 1's print,
// 25% of the median curve comes from ratios anchored on the pre-launch peak
// instead of on hour 1. If the case has no such peak, the blend is skipped.
//
// Spread: s_h is the weighted interquartile range of the peer log ratios at
// hour h, converted to a normal scale (IQR / 1.349), never below 0.15, and
// floored by 0.15 * sqrt(h - 1) — the house baseline's own random-walk floor —
// so uncertainty honestly grows with horizon and never collapses when only a
// few peers report an hour.
//
// Everything is deterministic (no clock, no randomness, files read in name
// order); the program never crashes on odd data — any failure falls back to
// the persistence baseline, which is always a valid forecast.
//
// Node 22, standard library only, no network.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Standard normal quantiles at the scored levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Hours to forecast. */
const HOURS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];

/** Similarity weights for one comparable launch. Tuned once, on leave-one-out
 *  runs over the comparables (never on the practice answers alone). */
const W_ROLLING = 3;
const W_CONSOLE = 2;
const W_SIMULTANEOUS = 0.25;
const W_EARLY_ACCESS = 0.25;
/** Scale-similarity length: exp(-|ln(h1_peer / p1)| / SCALE_LEN). */
const SCALE_LEN = 1;
/** Share of the median curve taken from the pre-launch-peak anchor. */
const PRE_PEAK_BLEND = 0.25;
/** Minimum log spread, and the random-walk floor per square-root hour. */
const S_MIN = 0.15;
const S_FLOOR_PER_SQRT_HOUR = 0.15;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly?: HourlyPoint[];
}

interface CaseFile {
  id?: string;
  day_start?: string;
  known?: Array<{ hour: number; viewers: number }>;
  forecast?: Array<{ hour: number }>;
  history_hourly?: HourlyPoint[];
  comparables?: Comparable[];
}

interface Peer {
  name: string;
  h1: number;
  prePeak: number;
  weight: number;
  /** Release-day hourly viewers, keyed by hour 1..24. */
  byHour: Map<number, number>;
}

const isFiniteNumber = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** Round to 2 decimals (half up), like the house baseline. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Weighted quantile of [value, weight] pairs (weights need not sum to 1). */
function weightedQuantile(pairs: Array<[number, number]>, q: number): number {
  const sorted = pairs.filter((p) => Number.isFinite(p[0]) && p[1] > 0).sort((a, b) => a[0] - b[0] || (a[1] - b[1]));
  let total = 0;
  for (const p of sorted) total += p[1];
  if (sorted.length === 0 || total <= 0) return NaN;
  const target = q * total;
  let acc = 0;
  for (const p of sorted) {
    acc += p[1];
    if (acc >= target) return p[0];
  }
  return sorted[sorted.length - 1][0];
}

const isRollingMidnight = (unlock: string): boolean => /midnight local/i.test(unlock);
const isSimultaneous = (unlock: string): boolean => /single global|simultaneous/i.test(unlock);
const isEarlyAccess = (unlock: string): boolean => !/no early access/i.test(unlock) && /early.access/i.test(unlock);
const isConsoleOnly = (platforms: string): boolean => !/PC/i.test(platforms);

/** Similarity weight of one comparable launch for a target with hour-1 print p1. */
function peerWeight(peer: { unlock: string; platforms: string; h1: number }, p1: number): number {
  let w = 1;
  if (isRollingMidnight(peer.unlock)) w *= W_ROLLING;
  if (isSimultaneous(peer.unlock)) w *= W_SIMULTANEOUS;
  if (isConsoleOnly(peer.platforms)) w *= W_CONSOLE;
  if (isEarlyAccess(peer.unlock)) w *= W_EARLY_ACCESS;
  w *= Math.exp(-Math.abs(Math.log(peer.h1 / p1)) / SCALE_LEN);
  return w;
}

/** Pull the usable launches out of the case's comparables: each needs a
 *  release-day hour-1 print; hours with no point are simply absent. */
function extractPeers(c: CaseFile, p1: number): Peer[] {
  const peers: Peer[] = [];
  const comps = Array.isArray(c.comparables) ? c.comparables : [];
  for (const comp of comps) {
    if (!comp || typeof comp.name !== 'string' || !Array.isArray(comp.hourly)) continue;
    const day = Date.parse(comp.day_start);
    if (!Number.isFinite(day)) continue;
    const byHour = new Map<number, number>();
    let prePeak = 0;
    for (const p of comp.hourly) {
      if (!p || !isFiniteNumber(p.viewers) || p.viewers <= 0) continue;
      const t = Date.parse(p.from);
      if (!Number.isFinite(t)) continue;
      const h = Math.round((t - day) / 3600000) + 1;
      if (h >= 1 && h <= 24) byHour.set(h, p.viewers);
      if (t >= day - 24 * 3600000 && t < day && p.viewers > prePeak) prePeak = p.viewers;
    }
    const h1 = byHour.get(1);
    if (h1 === undefined || h1 <= 0) continue; // no anchor: cannot use as a peer
    peers.push({
      name: comp.name,
      h1,
      prePeak,
      weight: peerWeight({ unlock: comp.unlock ?? '', platforms: comp.platforms ?? '', h1 }, p1),
      byHour,
    });
  }
  return peers;
}

/** Largest hourly count in the 24 h before the case's release day, from the
 *  case's own hourly history (0 when there is none). */
function targetPrePeak(c: CaseFile): number {
  const day = Date.parse(c.day_start ?? '');
  if (!Number.isFinite(day) || !Array.isArray(c.history_hourly)) return 0;
  let m = 0;
  for (const p of c.history_hourly) {
    if (!p || !isFiniteNumber(p.viewers)) continue;
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    if (t >= day - 24 * 3600000 && t < day && p.viewers > m) m = p.viewers;
  }
  return m;
}

/** Weighted-median log-ratio curve for hours 2..24, anchored on each peer's
 *  `anchor` value ('h1' or 'prePeak'). Missing entries come back as NaN. */
function anchorCurve(peers: Peer[], anchor: 'h1' | 'prePeak'): number[] {
  const curve: number[] = [];
  for (const h of HOURS) {
    const pairs: Array<[number, number]> = [];
    for (const peer of peers) {
      const v = peer.byHour.get(h);
      const a = anchor === 'prePeak' ? peer.prePeak : peer.h1;
      if (v === undefined || v <= 0 || !a || a <= 0) continue;
      pairs.push([Math.log(v / a), peer.weight]);
    }
    curve.push(pairs.length > 0 ? weightedQuantile(pairs, 0.5) : NaN);
  }
  return curve;
}

/** 3-point moving average in log space; endpoints average with their one neighbour. */
function smooth3(curve: number[]): number[] {
  return curve.map((v, i) => {
    const a = curve[Math.max(0, i - 1)];
    const b = v;
    const d = curve[Math.min(curve.length - 1, i + 1)];
    return Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(d) ? (a + b + d) / 3 : b;
  });
}

/** The persistence baseline (house baseline's shape): every hour stays at
 *  hour 1's print, spread widening like a random walk. Used as the fallback
 *  whenever the peer model has nothing to work with. */
function baselineQuantiles(p1: number): Array<{ hour: number; q: number[] }> {
  return HOURS.map((h) => ({
    hour: h,
    q: Z.map((z) => round2(clamp(p1 * Math.exp(z * S_FLOOR_PER_SQRT_HOUR * Math.sqrt(h - 1)), 1, 1e9))),
  }));
}

function forecastCase(c: CaseFile): Array<{ hour: number; q: number[] }> {
  const first = Array.isArray(c.known) ? c.known.find((k) => k && k.hour === 1) : undefined;
  const p1 = first && isFiniteNumber(first.viewers) && first.viewers > 0 ? first.viewers : 1;
  const hours = Array.isArray(c.forecast) && c.forecast.length > 0
    ? c.forecast.map((f) => f.hour).filter((h) => Number.isInteger(h) && h >= 2 && h <= 24)
    : HOURS;
  const peers = extractPeers(c, p1);
  if (peers.length < 3) return baselineQuantiles(p1).filter((r) => hours.includes(r.hour));

  // Median log-ratio curve anchored on hour 1. An hour no peer reports (a gap
  // in the data, not a real zero) inherits its nearest reported neighbour's
  // value instead of falling back to persistence (ratio 1.0).
  const rawMed = anchorCurve(peers, 'h1');
  let med = rawMed.map((v, i) => {
    if (Number.isFinite(v)) return v;
    for (let d = 1; d < rawMed.length; d++) {
      const lo = rawMed[i - d];
      const hi = rawMed[i + d];
      if (Number.isFinite(lo) && Number.isFinite(hi)) return (lo + hi) / 2;
      if (Number.isFinite(lo)) return lo;
      if (Number.isFinite(hi)) return hi;
    }
    return 0;
  });
  // Steady it with the pre-launch-peak anchor when the target shows one.
  const prePeak = targetPrePeak(c);
  if (prePeak > p1) {
    const alt = anchorCurve(peers, 'prePeak');
    const shift = Math.log(prePeak / p1);
    for (let i = 0; i < med.length; i++) {
      if (Number.isFinite(alt[i])) med[i] = (1 - PRE_PEAK_BLEND) * med[i] + PRE_PEAK_BLEND * (alt[i] + shift);
    }
  }
  med = smooth3(med);

  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const h of hours) {
    const i = HOURS.indexOf(h);
    const pairs: Array<[number, number]> = [];
    for (const peer of peers) {
      const v = peer.byHour.get(h);
      if (v === undefined || v <= 0) continue;
      pairs.push([Math.log(v / peer.h1), peer.weight]);
    }
    let s = (weightedQuantile(pairs, 0.75) - weightedQuantile(pairs, 0.25)) / 1.349;
    if (!Number.isFinite(s) || s <= 0) s = S_MIN;
    s = Math.max(S_MIN, s);
    s = Math.max(s, S_FLOOR_PER_SQRT_HOUR * Math.sqrt(h - 1));
    const m = i >= 0 ? med[i] : 0;
    let prev = 0;
    const q = Z.map((z) => {
      const v = clamp(round2(p1 * Math.exp(m + z * s)), 1, 1e9);
      const r = Math.max(prev, v);
      prev = r;
      return r;
    });
    rows.push({ hour: h, q });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let hours: Array<{ hour: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
    hours = forecastCase(c);
  } catch {
    // Unreadable case: still emit a validly-shaped forecast, never crash.
    hours = baselineQuantiles(1);
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
