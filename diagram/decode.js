// Position-level reasoning of the diagram recognition: from the class probabilities of the squares to the position.
// Identical glyphs of one diagram are one piece (units of squares that share their class), and a position follows the
// rules and looks like positions in books do: one king per side, at most eight pawns and none on the first or last
// rank, at most sixteen pieces per side, promoted pieces only for pawns that are gone, piece counts as in real games
// (two queens or three bishops of one side are rare), and a known opening fits better than the same position with a
// piece or two elsewhere. A beam search over the units finds the most probable positions; how the probability spreads
// over them gives the confidence of every square and of the whole board. No DOM (runs in Node for the benchmark).
import { OPENING_PLACEMENTS } from './openings.js';

export const CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
const IDX = Object.fromEntries(CLASSES.map((c, i) => [c, i]));
const ROLE_OF = [null, 'p', 'n', 'b', 'r', 'q', 'k', 'p', 'n', 'b', 'r', 'q', 'k'];
const isWhite = (c) => c >= 1 && c <= 6;
const isBlack = (c) => c >= 7;

/** how often a side has 0, 1, 2 … pieces of a kind (positions of real games and books); more is a promotion */
const COUNT_PRIOR = {
  q: [0.35, 0.6, 0.04, 0.01],
  r: [0.12, 0.28, 0.58, 0.02],
  b: [0.2, 0.4, 0.38, 0.02],
  n: [0.27, 0.38, 0.33, 0.02],
  p: [0.03, 0.03, 0.04, 0.07, 0.11, 0.17, 0.18, 0.2, 0.17],
  k: [0.03, 0.97],
};
const LOG_PRIOR = Object.fromEntries(Object.entries(COUNT_PRIOR).map(([k, a]) => [k, a.map(Math.log)]));
const BASE = { q: 1, r: 2, b: 2, n: 2 };
/** cost (log probability) of giving two clearly different glyphs of one colour the same piece */
const APART = 6;

/** the log prior of the piece counts of a position: counts[13] by class */
export function countPrior(counts, weight = 0.5) {
  let s = 0;
  for (const side of [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]) {
    const [P, Nn, B, R, Q, K] = side.map((c) => counts[c]);
    const n = { p: P, n: Nn, b: B, r: R, q: Q, k: K };
    for (const k of Object.keys(LOG_PRIOR)) {
      const a = LOG_PRIOR[k];
      s += weight * (n[k] < a.length ? a[n[k]] : a[a.length - 1] - 2 * (n[k] - a.length + 1));
    }
    // promoted pieces need pawns that are gone
    const extra = Math.max(0, Q - BASE.q) + Math.max(0, R - BASE.r) + Math.max(0, B - BASE.b) + Math.max(0, Nn - BASE.n);
    if (extra > 8 - P) s -= 8 * (extra - (8 - P));
  }
  return s;
}

/** hard limits while the search fills the board */
function violates(counts, limits) {
  for (const [lo, hi] of [[1, 6], [7, 12]]) {
    let total = 0;
    for (let c = lo; c <= hi; c++) total += counts[c];
    if (total > 16) return true;
    if (counts[lo] > 8) return true;                     // pawns
    if (counts[hi] > limits.kings) return true;          // kings
  }
  return false;
}

const expand = (pl) => pl.split('/').map((r) => r.replace(/\d/g, (d) => '.'.repeat(+d))).join('');
let OPENINGS = null;
/** the known placements as class indices per image square, seen from White's side and from Black's side */
function openings() {
  if (!OPENINGS) {
    OPENINGS = OPENING_PLACEMENTS.map((pl) => {
      const g = expand(pl);
      const white = Uint8Array.from(g, (ch) => IDX[ch]);
      const black = Uint8Array.from([...g].reverse(), (ch) => IDX[ch]);
      return { white, black };
    });
  }
  return OPENINGS;
}

/**
 * decodePosition(logp, units, opts) → { classes: [64], conf: [64], post: [64][13], boardConf, known, score }
 * logp: per square Float64Array(13), the log probabilities of the classes (squares outside the board units included)
 * units: [{ squares: [i…], fixed: class index | null }] covering every square once
 * opts: { orientation: 'white'|'black'|null, knownBonus, countWeight, beam, kings (largest number of kings per side) }
 */
export function decodePosition(logp, units, opts = {}) {
  const B = opts.beam || 300;
  const limits = { kings: opts.kings || 1 };
  // the log likelihood of each class for a unit, pawns never on the outer rows of the image
  const U = units.map((u) => {
    const L = new Float64Array(13);
    for (const i of u.squares) for (let c = 0; c < 13; c++) L[c] += logp[i][c];
    if (u.squares.some((i) => i < 8 || i >= 56)) { L[1] = -Infinity; L[7] = -Infinity; }
    let cands;
    if (u.fixed != null) cands = [u.fixed];
    else {
      const best = Math.max(...L);
      // classes more than e^8 (3000 times) less likely than the best one never matter
      cands = [...Array(13).keys()].filter((c) => L[c] > best - 8).sort((a, b) => L[b] - L[a]).slice(0, 5);
    }
    const sorted = cands.map((c) => L[c]);
    return { ...u, L, cands, margin: u.fixed != null ? Infinity : sorted.length > 1 ? sorted[0] - sorted[1] : Infinity };
  });
  // certain units first: the beam then branches only where it matters
  const order = U.map((u, k) => k).sort((a, b) => U[b].margin - U[a].margin);
  let beam = [{ score: 0, counts: new Int16Array(13), pick: new Int8Array(U.length).fill(-1) }];
  for (const k of order) {
    const u = U[k], n = u.squares.length;
    const next = [];
    for (const st of beam) {
      for (const c of u.cands) {
        if (!Number.isFinite(u.L[c])) continue;
        const counts = st.counts.slice();
        counts[c] += n;
        if (c && u.fixed == null && violates(counts, limits)) continue;
        const pick = st.pick.slice();
        pick[k] = c;
        // two clearly different glyphs of one colour are rarely the same piece
        let pen = 0;
        if (c && u.apart) for (const v of u.apart) if (pick[v] === c) pen += APART;
        next.push({ score: st.score + u.L[c] - pen, counts, pick });
      }
    }
    if (!next.length) {
      // nothing fits the limits (a fixed square broke them, or the board is no position): take the likeliest class
      for (const st of beam) { const c = u.cands[0]; const counts = st.counts.slice(); counts[c] += n; const pick = st.pick.slice(); pick[k] = c; next.push({ score: st.score + (Number.isFinite(u.L[c]) ? u.L[c] : -50), counts, pick }); }
    }
    next.sort((a, b) => b.score - a.score);
    beam = next.slice(0, B);
  }
  const cw = opts.countWeight ?? 0.5;
  for (const st of beam) st.total = st.score + countPrior(st.counts, cw);
  beam.sort((a, b) => b.total - a.total);
  // per square classes of a state
  const classesOf = (st) => {
    const out = new Array(64).fill(0);
    U.forEach((u, k) => { for (const i of u.squares) out[i] = st.pick[k]; });
    return out;
  };
  const sqScore = (cls) => { let s = 0; for (let i = 0; i < 64; i++) s += logp[i][cls[i]]; return s; };
  let best = classesOf(beam[0]);
  let bestTotal = beam[0].total;
  // known openings: one that differs in a few squares and is about as likely square by square wins, its position
  // being far more likely in a book than the same with a piece or two elsewhere
  let known = null;
  const bonus = opts.knownBonus ?? 5;
  if (bonus > 0) {
    const bestSq = sqScore(best) + countPrior(countsOf(best), cw);
    for (const o of openings()) {
      for (const side of opts.orientation ? [opts.orientation] : ['white', 'black']) {
        const g = o[side];
        let diff = 0;
        for (let i = 0; i < 64 && diff <= 6; i++) if (g[i] !== best[i]) diff++;
        if (diff === 0 || diff > 6) continue;
        // fixed squares are never overruled
        if (units.some((u) => u.fixed != null && u.squares.some((i) => g[i] !== u.fixed))) continue;
        const cls = Array.from(g);
        if (cls.some((c, i) => !Number.isFinite(logp[i][c]))) continue;
        const s = sqScore(cls) + countPrior(countsOf(cls), cw) + bonus;
        if (s > bestSq && (!known || s > known.s)) known = { s, cls, diff, side };
      }
    }
    if (known) { best = known.cls; bestTotal = known.s; }
  }
  // posterior over the beam (and the known opening): every state weighted by its probability
  const states = beam.map((st) => ({ cls: classesOf(st), total: st.total }));
  if (known) states.push({ cls: known.cls, total: known.s });
  const top = Math.max(...states.map((s) => s.total));
  let z = 0;
  for (const s of states) { s.w = Math.exp(s.total - top); z += s.w; }
  const post = Array.from({ length: 64 }, () => new Float64Array(13));
  for (const s of states) for (let i = 0; i < 64; i++) post[i][s.cls[i]] += s.w / z;
  const conf = best.map((c, i) => post[i][c]);
  let same = 0;
  for (const s of states) if (s.cls.every((c, i) => c === best[i])) same += s.w / z;
  return { classes: best, conf, post, boardConf: same, known: known ? { diff: known.diff, side: known.side } : null, score: bestTotal, rules: ruleBreaks(countsOf(best)) };
}

/** what a position breaks: 'kings' (not one king per side), 'pawns' (more than eight), 'promotions' (more extra
 *  pieces than pawns are gone), 'pieces' (more than sixteen on one side) */
export function ruleBreaks(counts) {
  const out = new Set();
  for (const [lo, hi] of [[1, 6], [7, 12]]) {
    const [P, Nn, B, R, Q, K] = [0, 1, 2, 3, 4, 5].map((k) => counts[lo + k]);
    if (K !== 1) out.add('kings');
    if (P > 8) out.add('pawns');
    if (P + Nn + B + R + Q + K > 16) out.add('pieces');
    const extra = Math.max(0, Q - 1) + Math.max(0, R - 2) + Math.max(0, B - 2) + Math.max(0, Nn - 2);
    if (extra > 8 - P) out.add('promotions');
  }
  return [...out];
}

function countsOf(cls) {
  const c = new Int16Array(13);
  for (const v of cls) c[v]++;
  return c;
}

export { isWhite, isBlack, ROLE_OF };
