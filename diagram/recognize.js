// Chess diagram recognition, classical and offline: board localisation (locate.js), square analysis
// (squares.js), shape features (features.js) and a nearest-template classification against pieces rendered from
// local piece sets (templates.js), corrected by the consistency inside the diagram (identical glyphs get the
// same piece) and by the rules (one king per side, no pawns on the first or last rank, at most eight pawns).
// recognize() takes a grey image (image.js) and returns the position as seen in the diagram plus the FEN
// placement after the orientation is decided (coordinates, pawn structure). reclassify() repeats the
// classification after corrections, with the corrected squares fixed and the book's templates added.
import { findBoard, refineGrid } from './locate.js';
import { analyzeSquares, N } from './squares.js';
import { pieceFeatures, FEATURE_GROUPS, FEATURE_LENGTH } from './features.js';
import { components, holes, dilate, histQuantile } from './image.js';
import { classify as classifyNet, diagramContext, setModel } from './net.js';
import { decodePosition, CLASSES, ROLE_OF, isBlack } from './decode.js';
import MODEL from './net-weights.js';

setModel(MODEL);

export const ROLES = ['k', 'q', 'r', 'b', 'n', 'p'];

/** template set: z-scaled feature rows with role, colour and the piece set they come from */
export function makeTemplateSet(items, scaleFrom = null, keepLda = false) {
  const L = FEATURE_LENGTH;
  let w = scaleFrom?.w;
  if (!w) {
    const mean = new Float64Array(L), sd = new Float64Array(L);
    for (const it of items) for (let i = 0; i < L; i++) mean[i] += it.feat[i];
    for (let i = 0; i < L; i++) mean[i] /= Math.max(1, items.length);
    for (const it of items) for (let i = 0; i < L; i++) sd[i] += (it.feat[i] - mean[i]) ** 2;
    // one scale per group (the mean spread of its dimensions): scaling every dimension by its own spread would
    // blow up the nearly constant ones, e.g. the topmost rows of the width profile
    w = new Float32Array(L);
    let o = 0;
    for (const [, len, wt] of FEATURE_GROUPS) {
      let g = 0;
      for (let i = 0; i < len; i++) g += Math.sqrt(sd[o + i] / Math.max(1, items.length));
      g /= len;
      for (let i = 0; i < len; i++) w[o + i] = wt / Math.sqrt(len) / (g + 0.02);
      o += len;
    }
  }
  const X = items.map((it) => { const v = new Float32Array(L); for (let i = 0; i < L; i++) v[i] = it.feat[i] * w[i]; return v; });
  const set = {
    items, X, w, lda: keepLda && scaleFrom?.lda ? scaleFrom.lda : lda(items.filter((it) => !it.book), X.filter((x, i) => !items[i].book)),
    /** the same set without the templates of some piece sets (benchmark: unseen piece sets) */
    without(...names) { const keep = items.filter((it) => !names.includes(it.set)); return makeTemplateSet(keep, set); },
    /** plus templates of one book (corrections), scaled like the rest */
    plus(extra) { return extra?.length ? makeTemplateSet([...items, ...extra.map((e) => ({ ...e, set: 'book', book: true }))], set, true) : set; },
  };
  return set;
}

/**
 * Linear discriminant over the roles that generalises across piece sets: the class means are the means of the
 * per-set class means, and the spread inside a class includes the differences between the sets, so the
 * discriminant leans on what all sets share (a pawn is small and round, a bishop tall and pointed). Shrunk
 * towards the diagonal; returns { a: [role] → weights, b: [role] → offset } or null.
 */
function lda(items, X) {
  if (items.length < 30) return null;
  const L = X[0].length;
  const sets = [...new Set(items.map((it) => it.set))];
  const means = {};
  for (const r of ROLES) {
    const m = new Float64Array(L);
    let n = 0;
    for (const st of sets) {
      const idx = items.map((it, i) => (it.role === r && it.set === st ? i : -1)).filter((i) => i >= 0);
      if (!idx.length) continue;
      const sm = new Float64Array(L);
      for (const i of idx) for (let k = 0; k < L; k++) sm[k] += X[i][k];
      for (let k = 0; k < L; k++) m[k] += sm[k] / idx.length;
      n++;
    }
    if (!n) return null;
    for (let k = 0; k < L; k++) m[k] /= n;
    means[r] = m;
  }
  const S = new Float64Array(L * L);
  for (let i = 0; i < items.length; i++) {
    const m = means[items[i].role], x = X[i];
    const d = new Float64Array(L);
    for (let k = 0; k < L; k++) d[k] = x[k] - m[k];
    for (let a = 0; a < L; a++) { const da = d[a]; if (!da) continue; for (let b = a; b < L; b++) S[a * L + b] += da * d[b]; }
  }
  for (let a = 0; a < L; a++) for (let b = a; b < L; b++) { S[a * L + b] /= items.length; S[b * L + a] = S[a * L + b]; }
  let tr = 0; for (let a = 0; a < L; a++) tr += S[a * L + a];
  const lam = 0.35, floor = (tr / L) * 0.05;
  for (let a = 0; a < L; a++) for (let b = 0; b < L; b++) S[a * L + b] *= a === b ? 1 : 1 - lam;
  for (let a = 0; a < L; a++) S[a * L + a] += floor;
  // Cholesky S = C Cᵀ, then solve S a_r = μ_r
  const C = new Float64Array(L * L);
  for (let i = 0; i < L; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = S[i * L + j];
      for (let k = 0; k < j; k++) sum -= C[i * L + k] * C[j * L + k];
      if (i === j) { if (sum <= 0) return null; C[i * L + i] = Math.sqrt(sum); } else C[i * L + j] = sum / C[j * L + j];
    }
  }
  const solve = (bv) => {
    const y = new Float64Array(L);
    for (let i = 0; i < L; i++) { let sum = bv[i]; for (let k = 0; k < i; k++) sum -= C[i * L + k] * y[k]; y[i] = sum / C[i * L + i]; }
    const xv = new Float64Array(L);
    for (let i = L - 1; i >= 0; i--) { let sum = y[i]; for (let k = i + 1; k < L; k++) sum -= C[k * L + i] * xv[k]; xv[i] = sum / C[i * L + i]; }
    return xv;
  };
  const a = {}, b = {};
  for (const r of ROLES) { a[r] = solve(means[r]); let q = 0; for (let k = 0; k < L; k++) q += a[r][k] * means[r][k]; b[r] = -0.5 * q; }
  // temperature: the typical margin of the right role over the best wrong one on the templates becomes 3 nats,
  // so that the discriminant gives probabilities, not certainties (the rules and the groups still have a say)
  const margins = [];
  for (let i = 0; i < items.length; i += 3) {
    const g = {};
    for (const r of ROLES) { let v = b[r]; for (let k = 0; k < L; k++) v += a[r][k] * X[i][k]; g[r] = v; }
    const right = g[items[i].role], wrong = Math.max(...ROLES.filter((r) => r !== items[i].role).map((r) => g[r]));
    margins.push(Math.abs(right - wrong));
  }
  margins.sort((x, y) => x - y);
  const temp = Math.max(1e-6, (margins[margins.length >> 1] || 1) / 3);
  return { a, b, temp };
}

/** templates from rendered boards with known pieces: { img, xs, ys, labels (64 chars, row-major), set } */
export function buildTemplates(boards) {
  const items = [];
  for (const b of boards) {
    const an = analyzeSquares(b.img, b.xs, b.ys, { arrows: false });
    const hmax = tallest(an.squares.filter((sq) => b.labels[sq.i] && b.labels[sq.i] !== '.' && sq.box));
    for (const sq of an.squares) {
      const ch = b.labels[sq.i];
      if (!ch || ch === '.' || !sq.box) continue;
      const black = ch === ch.toLowerCase();
      items.push({ role: ch.toLowerCase(), black, set: b.set, light: sq.light, feat: pieceFeatures(sq, black, hmax) });
    }
  }
  return makeTemplateSet(items);
}

/** height of the tallest pieces of a diagram (one bad silhouette does not count) */
export function tallest(squares) {
  const hs = squares.filter((s) => s.box).map((s) => s.box.h).sort((a, b) => b - a);
  if (!hs.length) return N * 0.9;
  return hs.length >= 2 ? Math.max(hs[1], 0.94 * hs[0]) : hs[0];
}

/**
 * Role probabilities of one square: the nearest template per role among the templates of the square's colour,
 * softmax over the distances, blended with the discriminant over all sets (which carries the decision when no
 * template is close: a piece set none of the templates comes from). A book's own corrections count more, and
 * among them '.' (a false piece the reader removed) is a role too. The nearest templates of the other colour
 * (compared with the features computed for that colour) tell how well the other colour would fit.
 */
function classify(tset, feat, featOther, black) {
  const L = FEATURE_LENGTH, w = tset.w;
  const xs = [feat, featOther].map((f) => { const x = new Float32Array(L); for (let i = 0; i < L; i++) x[i] = f[i] * w[i]; return x; });
  const best = {}, other = {};
  let bookNear = Infinity;
  for (let t = 0; t < tset.X.length; t++) {
    const it = tset.items[t];
    const same = it.role === '.' || it.black === black;
    const x = same ? xs[0] : xs[1], v = tset.X[t];
    let d = 0;
    for (let i = 0; i < L; i++) { const e = x[i] - v[i]; d += e * e; }
    if (it.book) { d *= 0.75; if (same && d < bookNear) bookNear = d; }
    const tab = same ? best : other;
    if (!(d >= tab[it.role])) tab[it.role] = d;
  }
  const roles = Object.keys(best);
  if (!roles.length) return { p: Object.fromEntries(ROLES.map((r) => [r, 1 / 6])), dist: best, dmin: Infinity, dOther: Infinity, bookNear };
  const dmin = Math.min(...roles.map((r) => best[r]));
  const dOther = Math.min(...Object.values(other), Infinity);
  const T = 0.18 * dmin + 0.08;
  const p = {};
  let sum = 0;
  for (const r of ROLES) { p[r] = r in best ? Math.exp(-(best[r] - dmin) / T) : 0; sum += p[r]; }
  if ('.' in best) { p['.'] = Math.exp(-(best['.'] - dmin) / T); sum += p['.']; }
  for (const r of Object.keys(p)) p[r] /= sum || 1;
  if (tset.lda) {
    const x = xs[0], g = {};
    for (const r of ROLES) { let v = tset.lda.b[r]; const a = tset.lda.a[r]; for (let i = 0; i < L; i++) v += a[i] * x[i]; g[r] = v; }
    const gm = Math.max(...ROLES.map((r) => g[r]));
    let s2 = 0; const q = {};
    for (const r of ROLES) { q[r] = Math.exp((g[r] - gm) / tset.lda.temp); s2 += q[r]; }
    const near = Math.exp(-dmin / 0.6);   // 1 for a template right there, towards 0 far from all of them
    let s3 = 0;
    for (const r of ROLES) { p[r] = Math.pow(p[r] + 1e-6, 0.5 + 0.5 * near) * Math.pow(q[r] / s2 + 1e-6, 1 - 0.6 * near); s3 += p[r]; }
    if ('.' in p) s3 += p['.'];
    for (const r of Object.keys(p)) p[r] /= s3 || 1;
  }
  // no role is ever ruled out completely: the rules and the groups may still need it
  let s4 = 0;
  for (const r of Object.keys(p)) { p[r] = Math.max(p[r], 1e-3); s4 += p[r]; }
  for (const r of Object.keys(p)) p[r] /= s4;
  return { p, dist: best, dmin, dOther, bookNear };
}

/**
 * recognize(img, { templates, hints, grid, net }) → {
 *   ok, error?, board: { xs, ys, skew }, grid: [8 strings, image rows from the top], orientation: 'white'|'black',
 *   orientationSource, placement (FEN field from White's side), squares: [{ r, c, ch, conf, alt }], confidence,
 *   uncertain: [square indices], known (a known opening was preferred), state (for reclassify) }
 * hints: { files: 'abcdefgh'|'hgfedcba' (coordinates read from a PDF text layer) }
 * analysis: the result of analyzeDiagram(img) when the caller has it already (training tools)
 * confidence: the probability that the whole position is right; conf of a square: that this square is right
 */
export function recognize(img, opts = {}) {
  const d = opts.analysis || analyzeDiagram(img, opts);
  if (!d.an) return { ok: d.ok, error: d.error, board: d.board };
  const { found, board, an } = d;
  const occupied = an.squares.filter((s) => s.occ);
  // colour: darkness inside and just inside the outline, split into two groups (Otsu) when the diagram shows
  // both colours clearly apart, else at the middle
  const colourT = splitColours(occupied.map((s) => s.darkness));
  const hmax = tallest(occupied);
  // the network looks at every square with some evidence; the others are paper or pattern
  const ctx = diagramContext(an);
  const look = new Set(an.squares.filter((s) => s.occ || s.area > 0.004 || s.arrow || s.rawShare > 0.004).map((s) => s.i));
  const net = opts.net === false ? null : classifyNet(an.squares, ctx, look);
  const entries = an.squares.map((s) => {
    const e = { i: s.i, r: s.r, c: s.c, light: s.light, occ: !!s.occ, net: net ? (net[s.i] ? Array.from(net[s.i].logits) : EMPTY_NET) : null };
    if (s.box) {
      e.black = s.darkness > colourT;
      e.colourConf = Math.min(1, Math.abs(s.darkness - colourT) / 0.25);
      e.feat = pieceFeatures(s, e.black, hmax);
      e.featOther = pieceFeatures(s, !e.black, hmax);   // for a colour corrected by hand
    }
    return e;
  });
  const orient = decideOrientationFromCoords(found, opts.hints);
  const state = { entries, orientation: orient?.orientation || null, orientationSource: orient?.source || null, board, arrows: an.arrows.length };
  return finish(state, opts.templates, new Map());
}

/** a square the network did not look at (no evidence of anything), as logits: empty with 99.8 % */
const EMPTY_NET = [Math.log(0.998), ...new Array(12).fill(Math.log(0.002 / 12))];

/**
 * The board and its squares: the board searched in the snip (or the given grid), a second pass on the grid, the
 * analysis of the squares. → { ok, error?, found, board: { xs, ys, skew, ok, tag }, an (squares.js) }
 */
export function analyzeDiagram(img, opts = {}) {
  // opts.grid: known lines (tests), else the board is searched
  const found = opts.grid ? { xs: opts.grid.xs, ys: opts.grid.ys, img, skew: 0, ok: true, tag: 'given' } : findBoard(img, { debug: opts.debug });
  if (!found) return { ok: false, error: 'no-board' };
  const board = { xs: found.xs, ys: found.ys, skew: found.skew, ok: found.ok, tag: found.tag };
  if (opts.locateOnly) return { ok: found.ok, board };
  if (!found.ok && !opts.force) return { ok: false, error: 'no-board', board };
  // second pass: the grid placed on the boundaries between empty squares, then the squares again; an axis takes
  // the new lines when the boundaries under them are clearly stronger (a frame can pull the first lattice off)
  let an = analyzeSquares(found.img, found.xs, found.ys, opts.grid ? {} : { arrows: false, quick: true });
  if (!opts.grid) {
    const g = refineGrid(found.img, found.xs, found.ys, an.squares.map((s) => !s.occ), an.patterned.dark || an.patterned.light);
    const p = (found.xs[8] - found.xs[0]) / 8;
    const take = (gain, a, b) => gain > 1.08 && Math.max(...a.map((v, k) => Math.abs(v - b[k]))) < p * 0.25;
    const xs = take(g.gainX, g.xs, found.xs) ? g.xs : found.xs, ys = take(g.gainY, g.ys, found.ys) ? g.ys : found.ys;
    found.xs = xs; found.ys = ys; board.xs = xs; board.ys = ys;
    an = analyzeSquares(found.img, xs, ys);
  }
  return { ok: true, found, board, an };
}

/** classification of the entries with templates `tset`; `fixed`: square index → character set by hand */
export function reclassify(state, tset, fixed = new Map()) { return finish(state, tset, fixed); }

/**
 * Settings of the decision, measured on generated diagrams (training/tune.js): the temperature of the
 * network, the weight of the templates next to it and of a book's own corrections when one of them is close, the
 * bonus of a known opening and the weight of the piece counts (decode.js), and the calibration of the confidence.
 */
const TUNE = { netT: 0.5, wTpl: 0.15, wBook: 1.5, bookNear: 1.2, knownBonus: 4, countWeight: 1.5, calibA: 0.304, calibB: -0.118, calibC: -1.841, sqA: 0.474, sqB: -0.309 };
/** for the tuning script only */
export function setTuning(t) { Object.assign(TUNE, t); }
export const tuning = () => ({ ...TUNE });
/** a square is shown as unsure (yellow) below this confidence */
const UNSURE = 0.8;

/** after a correction by hand the other squares keep their first reading unless their own evidence changed (the
 *  glyph group of the corrected square, a template of the book): one correction never moves an unrelated piece */
const STICK = 3;

/** template classifications of a state per template set (the same state is decided again after every correction) */
const TPL_CACHE = new WeakMap();

function finish(state, tset, fixed) {
  const E = state.entries;
  let cached = TPL_CACHE.get(state);
  if (!cached || cached.tset !== tset) { cached = { tset, tpl: E.map((e) => (tset && e.feat ? templateProbs(tset, e) : null)) }; TPL_CACHE.set(state, cached); }
  const tpl = cached.tpl;
  const logp = E.map((e, i) => squareLogp(e, tpl[i]));
  if (fixed.size && state.first) for (let i = 0; i < 64; i++) if (!fixed.has(i)) logp[i][state.first[i]] += STICK;
  const units = glyphUnits(E, tpl, logp, tset, fixed);
  const dec = decodePosition(logp, units, { orientation: state.orientation, knownBonus: TUNE.knownBonus, countWeight: TUNE.countWeight });
  if (!fixed.size) state.first = dec.classes.slice();
  const sq = (p) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return 1 / (1 + Math.exp(-(TUNE.sqA * Math.log(p / (1 - p)) + TUNE.sqB))); };
  const out = E.map((e, i) => {
    const c = dec.classes[i], post = dec.post[i];
    const top = [...post.keys()].sort((a, b) => post[b] - post[a]).slice(0, 3).filter((k) => k === c || post[k] > 0.001);
    return { ...e, ch: CLASSES[c], conf: sq(dec.conf[i]), alt: top.map((k) => [CLASSES[k], +post[k].toFixed(3)]) };
  });
  for (const [i, ch] of fixed) { out[i].ch = ch; out[i].conf = 1; }
  const grid = [];
  for (let r = 0; r < 8; r++) grid.push(out.slice(r * 8, r * 8 + 8).map((o) => o.ch).join(''));
  const orient = state.orientation ? { orientation: state.orientation, source: state.orientationSource } : orientationFromPieces(out);
  const pieceSq = (o) => o.ch !== '.' || o.occ || (o.net && o.net[0] < 0.9);
  return {
    ok: true, board: state.board, grid, orientation: orient.orientation, orientationSource: orient.source,
    placement: toPlacement(grid, orient.orientation), confidence: calibrated(dec), rawConfidence: dec.boardConf,
    squares: out.map((o) => ({ r: o.r, c: o.c, ch: o.ch, conf: o.conf, alt: o.alt })),
    uncertain: out.filter((o) => !fixed.has(o.i) && pieceSq(o) && o.conf < UNSURE).map((o) => o.i),
    known: dec.known, rules: dec.rules, state,
  };
}

/** the probability that the whole position is right, from the decoder's spread over the positions it found, mapped
 *  onto what the benchmark measured (training/bench.js, calibration table) */
function calibrated(dec) {
  const p = Math.min(1 - 1e-6, Math.max(1e-6, dec.boardConf));
  // a position that breaks the rules (no king, too many promoted pieces …) is rarely read right
  const z = TUNE.calibA * Math.log(p / (1 - p)) + TUNE.calibB + (dec.rules.length ? TUNE.calibC : 0);
  return 1 / (1 + Math.exp(-z));
}

/** the template classification of a square: role probabilities, its colour and how close a book's template is */
function templateProbs(tset, e) {
  let black = e.black, colourConf = e.colourConf, feat = e.feat, featOther = e.featOther;
  let c = classify(tset, feat, featOther, black);
  // an unclear colour follows the templates when those of the other colour fit clearly better
  if (colourConf < 0.7 && c.dOther < 0.7 * c.dmin) {
    black = !black; [feat, featOther] = [featOther, feat]; colourConf = 0.5;
    c = classify(tset, feat, featOther, black);
  }
  return { p: c.p, black, colourConf, dmin: c.dmin, bookNear: c.bookNear, feat };
}

/**
 * Log probabilities of the 13 classes of one square: the network's, and the templates' over the pieces (they know
 * which piece a glyph is, not whether a glyph is there; empty only from a book's correction). Without the network
 * the silhouette decides whether there is a piece.
 */
function squareLogp(e, t) {
  const L = new Float64Array(13);
  if (e.net) {
    // log softmax of the network's logits at the tuned temperature
    let m = -Infinity; for (const v of e.net) if (v > m) m = v;
    let z = 0; for (const v of e.net) z += Math.exp((v - m) / TUNE.netT);
    for (let c = 0; c < 13; c++) L[c] = Math.max(Math.log(1e-6), (e.net[c] - m) / TUNE.netT - Math.log(z));
  }
  else if (t) { const occ = e.occ ? 0.95 : 0.3; L[0] = Math.log(1 - occ); for (let c = 1; c < 13; c++) L[c] = Math.log(occ / 12); }
  else { L[0] = Math.log(e.occ ? 0.1 : 0.98); for (let c = 1; c < 13; c++) L[c] = Math.log((e.occ ? 0.9 : 0.02) / 12); }
  if (!t) return L;
  const w = e.net ? (t.bookNear < TUNE.bookNear ? TUNE.wBook : TUNE.wTpl) : 1;
  const pb = t.black ? 0.5 + 0.5 * t.colourConf : 0.5 - 0.5 * t.colourConf;
  const none = t.p['.'] || 0;
  for (let c = 1; c < 13; c++) {
    const q = (1 - none) * (t.p[ROLE_OF[c]] || 0) * (isBlack(c) ? pb : 1 - pb);
    L[c] += w * Math.log(q + 1e-4);
  }
  L[0] += w * Math.log(none ? none + 1e-3 : 1 / 12);
  return L;
}

/** how many identical pieces of one colour a position holds at most (promotions aside) */
/** feature distance below which two pieces of one colour on the same kind of square are one glyph, and above
 *  which they are clearly two different glyphs (measured on the benchmark: the same glyph stays below 1 in nine
 *  of ten pairs even in scans, two different glyphs lie above 3 in 99 of 100) */
const SAME_GLYPH = 1.0, OTHER_GLYPH = 4.0;

/**
 * Identical glyphs in one diagram are the same piece: squares of one colour whose features are much closer to each
 * other than to any template form a unit that takes one class (three "bishops" that look exactly alike are pawns).
 * The same glyph on a light and on a dark square differs a little (hatching or grey along the outline), so the
 * groups of one colour are joined across the two kinds of squares when they are each other's nearest group there
 * and much closer to each other than to any other glyph on their own kind of square. The other way round, a book
 * prints every piece with one glyph: two clearly different glyphs of one colour on the same kind of square are
 * different pieces (the decoder makes them pay for the same class). A square set by hand fixes its whole group.
 * → [{ squares, fixed, apart: [unit indices] }] covering all 64 squares
 */
function glyphUnits(E, tpl, logp, tset, fixed) {
  const best = (i) => { let b = 0; for (let c = 1; c < 13; c++) if (logp[i][c] > logp[i][b]) b = c; return b; };
  const occ = [];
  for (const e of E) {
    if (fixed.has(e.i)) {
      const ch = fixed.get(e.i);
      if (ch === '.' || !e.feat) continue;
      const black = ch === ch.toLowerCase();
      occ.push({ i: e.i, fixed: ch, black, light: e.light, feat: black === e.black ? e.feat : e.featOther, dmin: 0 });
      continue;
    }
    if (!e.feat || !tset) continue;
    const c = best(e.i);
    if (!c) continue;
    const black = isBlack(c);
    const t = tpl[e.i];
    occ.push({ i: e.i, fixed: null, black, light: e.light, feat: black === e.black ? e.feat : e.featOther, dmin: t ? t.dmin : 1 });
  }
  const units = [];
  const inUnit = new Set();
  if (tset && occ.length) {
    const w = tset.w, L = FEATURE_LENGTH;
    const dist = (a, b) => { let d = 0; for (let i = 0; i < L; i++) { const e = (a[i] - b[i]) * w[i]; d += e * e; } return d; };
    const parent = occ.map((o, i) => i);
    const root = (i) => (parent[i] === i ? i : (parent[i] = root(parent[i])));
    for (let a = 0; a < occ.length; a++) for (let b = a + 1; b < occ.length; b++) {
      const o = occ[a], q = occ[b];
      if (o.black !== q.black || (o.fixed && q.fixed && o.fixed !== q.fixed)) continue;
      if (dist(o.feat, q.feat) < Math.max(SAME_GLYPH, 0.35 * Math.min(o.dmin, q.dmin) + 0.15)) parent[root(a)] = root(b);
    }
    const collect = () => {
      const m = new Map();
      occ.forEach((o, i) => { const r = root(i); if (!m.has(r)) m.set(r, []); m.get(r).push(i); });
      return [...m.values()].map((idx) => {
        const mean = new Float32Array(L);
        for (const i of idx) for (let k = 0; k < L; k++) mean[k] += occ[i].feat[k] / idx.length;
        return { idx, mean, black: occ[idx[0]].black, light: idx.some((i) => occ[i].light), dark: idx.some((i) => !occ[i].light) };
      });
    };
    // across the kinds of squares: mutual nearest groups, closer than half the distance to the nearest other glyph
    // and close compared with the templates
    const gs = collect();
    const D = gs.map((g) => gs.map((h) => (g === h || g.black !== h.black ? Infinity : dist(g.mean, h.mean))));
    const shares = (g, h) => (g.light && h.light) || (g.dark && h.dark);
    const own = gs.map((g, a) => Math.min(...gs.map((h, b) => (shares(g, h) ? D[a][b] : Infinity))));
    const nearestAcross = (a) => { let b0 = -1; gs.forEach((h, b) => { if (D[a][b] < Infinity && !shares(gs[a], h) && (b0 < 0 || D[a][b] < D[a][b0])) b0 = b; }); return b0; };
    const dminOf = (g) => Math.min(...g.idx.map((i) => occ[i].dmin));
    const fixedOf = (g) => g.idx.map((i) => occ[i].fixed).find(Boolean) || null;
    gs.forEach((g, a) => {
      const b = nearestAcross(a);
      if (b < 0 || b < a || nearestAcross(b) !== a) return;
      const h = gs[b], scale = Math.min(own[a], own[b]);
      if (fixedOf(g) && fixedOf(h) && fixedOf(g) !== fixedOf(h)) return;
      if (scale < Infinity && D[a][b] < 0.5 * scale && D[a][b] < 0.8 * Math.min(dminOf(g), dminOf(h)) + 0.4) parent[root(g.idx[0])] = root(h.idx[0]);
    });
    const groups = collect();
    groups.forEach((g) => {
      const fx = g.idx.map((i) => occ[i].fixed).filter(Boolean);
      const squares = g.idx.map((i) => occ[i].i);
      squares.forEach((i) => inUnit.add(i));
      units.push({ squares, fixed: fx.length ? CLASSES.indexOf(fx[0]) : null, group: g });
    });
    // different glyphs, different pieces: groups of one colour clearly apart on a kind of square both stand on
    const meanOn = (g, light) => {
      const idx = g.idx.filter((i) => occ[i].light === light);
      if (!idx.length) return null;
      const m = new Float32Array(L);
      for (const i of idx) for (let k = 0; k < L; k++) m[k] += occ[i].feat[k] / idx.length;
      return m;
    };
    const means = units.map((u) => [meanOn(u.group, true), meanOn(u.group, false)]);
    units.forEach((u) => { u.apart = []; });
    for (let a = 0; a < units.length; a++) for (let b = a + 1; b < units.length; b++) {
      if (units[a].group.black !== units[b].group.black) continue;
      for (const k of [0, 1]) {
        if (means[a][k] && means[b][k] && dist(means[a][k], means[b][k]) > OTHER_GLYPH) { units[a].apart.push(b); units[b].apart.push(a); break; }
      }
    }
    units.forEach((u) => { delete u.group; });
  }
  for (const e of E) {
    if (inUnit.has(e.i)) continue;
    units.push({ squares: [e.i], fixed: fixed.has(e.i) ? CLASSES.indexOf(fixed.get(e.i)) : null, apart: [] });
  }
  return units;
}

/** FEN placement (from White's side) of an image grid shown with `orientation` at the bottom */
export function toPlacement(grid, orientation) {
  const std = orientation === 'black' ? grid.map((row) => row.split('').reverse().join('')).reverse() : grid;
  return std.map((row) => row.replace(/\.+/g, (m) => String(m.length))).join('/');
}

/** threshold between white and black pieces: Otsu on the darkness values, kept between 0.3 and 0.7 */
function splitColours(vals) {
  if (vals.length < 3) return 0.5;
  const v = vals.slice().sort((a, b) => a - b);
  let best = -1, t = 0.5;
  for (let i = 1; i < v.length; i++) {
    const a = v.slice(0, i), b = v.slice(i);
    const ma = a.reduce((s, x) => s + x, 0) / a.length, mb = b.reduce((s, x) => s + x, 0) / b.length;
    const between = a.length * b.length * (mb - ma) ** 2;
    if (between > best) { best = between; t = (v[i - 1] + v[i]) / 2; }
  }
  const lo = v.filter((x) => x <= t), hi = v.filter((x) => x > t);
  const mlo = lo.reduce((s, x) => s + x, 0) / lo.length, mhi = hi.reduce((s, x) => s + x, 0) / hi.length;
  // two clear groups only when their means lie well apart
  if (!lo.length || !hi.length || mhi - mlo < 0.3) return 0.5;
  return Math.max(0.3, Math.min(0.7, t));
}

/** orientation from coordinates (a PDF text layer, or the letters and digits next to the board), or null */
function decideOrientationFromCoords(found, hints = {}) {
  if (hints?.files === 'abcdefgh') return { orientation: 'white', source: 'text' };
  if (hints?.files === 'hgfedcba') return { orientation: 'black', source: 'text' };
  const coords = readCoordinates(found);
  return coords ? { orientation: coords, source: 'coordinates' } : null;
}

/** without coordinates: White's pawns stand below Black's in a diagram from White's side; White by default (the
 *  kings say nothing: in an endgame White's king often stands at the top) */
function orientationFromPieces(out) {
  const rows = { w: [], b: [] };
  for (const o of out) if (o.ch === 'P') rows.w.push(o.r); else if (o.ch === 'p') rows.b.push(o.r);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  if (rows.w.length && rows.b.length) {
    const d = mean(rows.w) - mean(rows.b);
    if (d < -0.8) return { orientation: 'black', source: 'pawns' };
    if (d > 0.4) return { orientation: 'white', source: 'pawns' };
  }
  return { orientation: 'white', source: 'default' };
}

/** the coordinate labels next to the board, if there are any: 'white', 'black' or null */
function readCoordinates(found) {
  const img = found.img, { xs, ys } = found;
  const p = (xs[8] - xs[0]) / 8;
  // dark connected components in a strip, in snip coordinates: ink below a threshold between the darkest pixels
  // of the strip and its paper, the parts of a thin letter joined by a one-pixel dilation (box and holes come from
  // the ink itself); whatever touches the strip border (a frame, text running out of the strip) is left out
  const strip = (sx, sy, sw, sh) => {
    sx = Math.round(sx); sy = Math.round(sy); sw = Math.max(1, Math.round(sw)); sh = Math.max(1, Math.round(sh));
    const v = new Float32Array(sw * sh).fill(1);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const X = sx + x, Y = sy + y;
      if (X >= 0 && Y >= 0 && X < img.w && Y < img.h) v[y * sw + x] = img.d[Y * img.w + X];
    }
    const lo = histQuantile(v, 0.005), hi = histQuantile(v, 0.5);
    if (hi - lo < 0.25) return [];
    const t = lo + 0.6 * (hi - lo);
    const m = new Uint8Array(sw * sh);
    for (let j = 0; j < sw * sh; j++) m[j] = v[j] < t ? 1 : 0;
    const { labels, comps } = components(dilate(m, sw, sh, 1), sw, sh);
    const box = comps.map(() => ({ n: 0, x0: sw, y0: sh, x1: -1, y1: -1 }));
    for (let j = 0; j < sw * sh; j++) {
      if (!m[j]) continue;
      const b = box[labels[j] - 1], x = j % sw, y = (j - x) / sw;
      b.n++; if (x < b.x0) b.x0 = x; if (x > b.x1) b.x1 = x; if (y < b.y0) b.y0 = y; if (y > b.y1) b.y1 = y;
    }
    return comps.map((c, k) => ({ id: k + 1, n: box[k].n, x0: box[k].x0 + sx, x1: box[k].x1 + sx, y0: box[k].y0 + sy, y1: box[k].y1 + sy,
      lx0: box[k].x0, ly0: box[k].y0, lx1: box[k].x1, ly1: box[k].y1, labels, m, sw, edge: c.x0 === 0 || c.y0 === 0 || c.x1 === sw - 1 || c.y1 === sh - 1 }))
      .filter((g) => g.n >= 4 && !g.edge && g.x1 - g.x0 < p * 0.6 && g.y1 - g.y0 < p * 0.7 && g.y1 - g.y0 > p * 0.1);
  };
  let votes = 0;
  // files: the first line of text under the board, one letter per column. In "a b c d e f g h" the ascenders of
  // b, d, f and h stand above the x-height of a, c, e and g; in "h g f e d c b a" it is the other way round
  const below = strip(xs[0] - p * 0.2, ys[8] + p * 0.05, p * 8.4, p * 0.8);
  if (below.length) {
    const top0 = Math.min(...below.map((g) => g.y0));
    const line = below.filter((g) => g.y0 < top0 + p * 0.25);
    const cols = Array.from({ length: 8 }, (_, k) => line.filter((g) => Math.abs((g.x0 + g.x1) / 2 - (xs[k] + xs[k + 1]) / 2) < p * 0.3));
    if (cols.filter((c) => c.length === 1).length >= 6) {
      const tops = cols.map((c) => (c.length === 1 ? c[0].y0 : null));
      const mean = (ks) => { const t = ks.map((k) => tops[k]).filter((y) => y != null); return t.length >= 2 ? t.reduce((a, b) => a + b, 0) / t.length : null; };
      const even = mean([0, 2, 4, 6]), odd = mean([1, 3, 5, 7]);
      const hs = cols.filter((c) => c.length === 1).map((c) => c[0].y1 - c[0].y0).sort((a, b) => a - b);
      const hMed = hs[hs.length >> 1];
      if (even != null && odd != null) {
        if (even - odd > 0.15 * hMed) votes++;          // a … h
        else if (odd - even > 0.15 * hMed) votes--;     // h … a
      }
    }
  }
  // ranks: the label next to the board in each row; the eight has two holes and the six one, so the holes lie in
  // the upper half for "8 … 1" and in the lower half for "1 … 8"
  const left = strip(xs[0] - p * 0.95, ys[0] - p * 0.1, p * 0.9, p * 8.2);
  const rows = Array.from({ length: 8 }, (_, k) => left.filter((g) => Math.abs((g.y0 + g.y1) / 2 - (ys[k] + ys[k + 1]) / 2) < p * 0.3).sort((a, b) => b.x1 - a.x1).slice(0, 1));
  if (rows.filter((g) => g.length === 1).length >= 6) {
    const holesOf = (g) => {
      const w = g.lx1 - g.lx0 + 3, h = g.ly1 - g.ly0 + 3;
      const m = new Uint8Array(w * h);
      for (let y = g.ly0; y <= g.ly1; y++) for (let x = g.lx0; x <= g.lx1; x++) { const j = y * g.sw + x; if (g.labels[j] === g.id && g.m[j]) m[(y - g.ly0 + 1) * w + (x - g.lx0 + 1)] = 1; }
      return holes(m, w, h);
    };
    const hs = rows.map((g) => (g.length ? Math.min(2, holesOf(g[0])) : 0));
    const upper = hs.slice(0, 4).reduce((a, b) => a + b, 0), lower = hs.slice(4).reduce((a, b) => a + b, 0);
    if (upper - lower >= 2) votes++;          // 8 above
    else if (lower - upper >= 2) votes--;     // 8 below
  }
  if (votes > 0) return 'white';
  if (votes < 0) return 'black';
  return null;
}
