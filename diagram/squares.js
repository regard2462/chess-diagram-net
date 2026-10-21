// Square analysis: from the located grid to one normalised record per square. All squares of one colour look
// alike when empty, and most of them are empty, so the squares describe their own background. Every square is cut
// out with a margin; the squares of a patterned colour (hatching, stippling) are aligned to each other by a shift of
// a fraction of a pixel, found on all their pixels with a robust measure (the pixels of a piece cost the same at any
// shift, so the background decides; hatching repeats, so the shift also absorbs a grid that is a pixel off or a scan
// that stretches), and the pixelwise median of the aligned squares is the empty square of that colour. A piece is
// where a square differs from it: a black body, an outline, and on hatched or stippled squares the lines a white
// body hides. Arrows and lines drawn over the board are found on that evidence across the whole board (arrows.js)
// and taken out of it. The silhouette is the evidence closed and filled (an outline that the square's bottom edge
// cuts off is closed along it first). The ink inside it (dark pixels) tells white from black pieces and carries the
// inner details. Every square also comes cleaned for the classifier: the background (paper, hatching, arrows) white,
// the piece as it is printed.
import { cell, blur, boxBlur, gray, fillHoles, dilate, erode, components, percentile } from './image.js';
import { findArrows } from './arrows.js';

export const N = 48;          // normalised square size
const S = 6;                  // largest alignment shift (patterned squares: a full period of coarse stippling)
const K = S + 1;              // margin of the cut-out
const NE = N + 2 * K;         // size of the cut-out
const NH = N;                 // height of the evidence window (the square)
const M = 3;                  // border at the outer edge of the board that never belongs to a piece (frames)
export const SHIFT = S;
const CP = 7;                 // size of the corner patches (pattern spread)
const CORNERS = [[2, 2], [N - 2 - CP, 2], [2, N - 2 - CP], [N - 2 - CP, N - 2 - CP]];   // right in the corners: wide piece bases reach 15 % from the side

/** an N wide, `height` tall window of a cut-out (top left at the square's top left), shifted by (dx, dy); a shift of
 *  a fraction of a pixel interpolates between the pixels of the cut-out */
function windowOf(ext, dx, dy, height = N) {
  const out = { w: N, h: height, d: new Float32Array(N * height) };
  const ix = Math.floor(dx), iy = Math.floor(dy), fx = dx - ix, fy = dy - iy;
  if (!fx && !fy) {
    for (let y = 0; y < height; y++) {
      const sy = y + K + dy, sx = K + dx;
      out.d.set(ext.d.subarray(sy * NE + sx, sy * NE + sx + N), y * N);
    }
    return out;
  }
  const cl = (v) => (v < 0 ? 0 : v > NE - 1 ? NE - 1 : v);
  for (let y = 0; y < height; y++) {
    const r0 = cl(y + K + iy) * NE, r1 = cl(y + K + iy + 1) * NE;
    for (let x = 0; x < N; x++) {
      const x0 = cl(x + K + ix), x1 = cl(x + K + ix + 1);
      out.d[y * N + x] = (ext.d[r0 + x0] * (1 - fx) + ext.d[r0 + x1] * fx) * (1 - fy) + (ext.d[r1 + x0] * (1 - fx) + ext.d[r1 + x1] * fx) * fy;
    }
  }
  return out;
}

/** robust difference of a cut-out at shift (dx, dy) against a template: the mean absolute difference, each pixel
 *  capped at `cap` (a piece costs the same at any shift), over every `step`-th pixel of the inner square */
function robustCost(ext, dx, dy, tpl, cap, step, r = 3, c = 3) {
  const ix = Math.floor(dx), iy = Math.floor(dy), fx = dx - ix, fy = dy - iy;
  const frac = fx || fy;
  // only what lies inside the square under this shift counts (beyond it is the neighbour, or at the outer edge of
  // the board a frame), and not the band along the board's edge
  const iw = inside(r, c, dx, dy, 2);
  const ya = Math.max(iw.ya, Math.ceil(-dy)), yb = Math.min(iw.yb, Math.floor(N - dy)), xa = Math.max(iw.xa, Math.ceil(-dx)), xb = Math.min(iw.xb, Math.floor(N - dx));
  let e = 0, n = 0;
  const src = ext.d, t = tpl.d;
  if (frac) {
    const gx = 1 - fx, gy = 1 - fy;
    for (let y = ya; y < yb; y += step) {
      const r0 = (y + K + iy) * NE, r1 = r0 + NE;
      for (let x = xa; x < xb; x += step) {
        const sx = x + K + ix;
        const v = (src[r0 + sx] * gx + src[r0 + sx + 1] * fx) * gy + (src[r1 + sx] * gx + src[r1 + sx + 1] * fx) * fy;
        const d = Math.abs(v - t[y * N + x]);
        e += d < cap ? d : cap;
        n++;
      }
    }
  } else {
    for (let y = ya; y < yb; y += step) {
      const r0 = (y + K + iy) * NE + K + ix, rt = y * N;
      for (let x = xa; x < xb; x += step) {
        const d = Math.abs(src[r0 + x] - t[rt + x]);
        e += d < cap ? d : cap;
        n++;
      }
    }
  }
  return e / n;
}

/** what a shift costs in the alignment: of two equally good shifts along a hatch line the smaller wins, and beyond
 *  three pixels (only coarse stippling needs that) a shift has to fit clearly better: a square that a black piece
 *  almost fills leaves too little of its pattern to justify a large one */
const shiftCost = (dx, dy) => 0.003 * (Math.abs(dx) + Math.abs(dy)) + 0.012 * (Math.max(0, Math.abs(dx) - 3) + Math.max(0, Math.abs(dy) - 3));

/** the part of a window (shifted by dx, dy) that lies inside the board and away from its outer edge (a frame), with
 *  an extra margin `pad` → { ya, yb, xa, xb } (half-open) */
function inside(r, c, dx, dy, pad = 0) {
  return {
    ya: r === 0 ? Math.max(pad, M - Math.floor(dy)) : pad, yb: r === 7 ? Math.min(N - pad, N - M - Math.ceil(dy)) : N - pad,
    xa: c === 0 ? Math.max(pad, M - Math.floor(dx)) : pad, xb: c === 7 ? Math.min(N - pad, N - M - Math.ceil(dx)) : N - pad,
  };
}

/** the shift that brings a patterned square onto the template: whole pixels first, then quarters of a pixel; of
 *  equally good shifts the smallest wins (hatching repeats along its lines, and a large shift pulls a strip of the
 *  neighbour into view) */
function bestShift(ext, tpl, cap, r, c, fine = true) {
  const y0 = -S, y1 = S, x0 = -S, x1 = S;
  // every whole shift on every fourth pixel, the best few again on every second one
  const coarse = [];
  for (let dy = y0; dy <= y1; dy++) for (let dx = x0; dx <= x1; dx++) {
    coarse.push({ dx, dy, e: robustCost(ext, dx, dy, tpl, cap, 4, r, c) + shiftCost(dx, dy) - (!dx && !dy ? 0.002 : 0) });
  }
  coarse.sort((a, b) => a.e - b.e);
  let best = null;
  for (const o of coarse.slice(0, 6)) {
    const e = robustCost(ext, o.dx, o.dy, tpl, cap, 2, r, c) + shiftCost(o.dx, o.dy) - (!o.dx && !o.dy ? 0.002 : 0);
    if (!best || e < best.e) best = { dx: o.dx, dy: o.dy, e };
  }
  if (!fine) return [best.dx, best.dy];
  // half a pixel, then a quarter, around the best whole shift
  let cur = { dx: best.dx, dy: best.dy, e: robustCost(ext, best.dx, best.dy, tpl, cap, 2, r, c) };
  for (const step of [0.5, 0.25]) {
    const cx = cur.dx, cy = cur.dy;
    for (let oy = -step; oy <= step; oy += step) for (let ox = -step; ox <= step; ox += step) {
      if (!ox && !oy) continue;
      const dx = cx + ox, dy = cy + oy;
      if (dx < x0 - 0.5 || dx > x1 + 0.5 || dy < y0 - 0.5 || dy > y1 + 0.5 || Math.abs(dx) > S - 0.25 || Math.abs(dy) > S - 0.25) continue;
      const e = robustCost(ext, dx, dy, tpl, cap, 2, r, c) + shiftCost(dx, dy) - shiftCost(cx, cy);
      if (e < cur.e) cur = { dx, dy, e };
    }
  }
  return [cur.dx, cur.dy];
}

/** pixelwise median of windows */
function medianOf(wins) {
  const n = wins[0].d.length;
  const out = { w: wins[0].w, h: wins[0].h, d: new Float32Array(n) };
  const vals = new Float32Array(wins.length);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < wins.length; k++) vals[k] = wins[k].d[i];
    vals.sort();
    out.d[i] = vals[vals.length >> 1];
  }
  return out;
}

/** the empty square of a flat colour has nothing inside it that only some squares show: what the median of the
 *  emptiest squares kept there (an arrow that runs through many squares of that colour at the same place, along a
 *  diagonal) becomes the colour's level, unless the median of all squares of the colour shows it too (a strip of the
 *  neighbour that every square shows when the grid is a little off, a grid line); the edges stay as they are */
function flatInside(t, everyone) {
  const v = [];
  for (let y = 4; y < N - 4; y += 2) for (let x = 4; x < N - 4; x += 2) v.push(t.d[y * N + x]);
  v.sort((a, b) => a - b);
  const level = v[v.length >> 1];
  for (let y = 3; y < N - 3; y++) for (let x = 3; x < N - 3; x++) {
    const j = y * N + x;
    if (Math.abs(t.d[j] - level) > 0.04 && Math.abs(everyone.d[j] - level) <= 0.04) t.d[j] = level;
  }
  return t;
}

/** mean absolute difference of the inner part of two windows (3 pixels from the edges) */
function innerError(a, b) {
  let e = 0, n = 0;
  for (let y = 3; y < N - 3; y++) for (let x = 3; x < N - 3; x++) { e += Math.abs(a.d[y * N + x] - b.d[y * N + x]); n++; }
  return e / n;
}

/** the share of the inner part of a window that differs clearly from its own median: pieces, not paper */
function content(win) {
  const v = [];
  for (let y = 3; y < N - 3; y += 2) for (let x = 3; x < N - 3; x += 2) v.push(win.d[y * N + x]);
  v.sort((a, b) => a - b);
  const m = v[v.length >> 1];
  let n = 0;
  for (const x of v) if (Math.abs(x - m) > 0.25) n++;
  return n / v.length;
}

/**
 * The empty square of one colour and each square's alignment to it. The empty square is the pixelwise median of
 * the squares that look emptiest (whole squares, not only their corners: with pieces on half of the squares a
 * median over all of them would keep a ghost of the pieces in the middle). Patterned squares are aligned to one
 * empty-looking square first, then to the median, three times.
 */
function alignClass(exts, pos, contrast, quick = false) {
  const keepN = Math.max(3, Math.round(exts.length * 0.4));
  const plainWins = exts.map((e) => windowOf(e, 0, 0));
  // a flat colour needs no alignment (and a piece near a corner would only pull the square astray); whether the
  // colour is patterned shows in each square's corners (the plain median of hatching out of phase is flat)
  const spreads = plainWins.map((w) => spread(w)).sort((a, b) => a - b);
  if (spreads[spreads.length >> 1] <= 0.06) {
    let order = plainWins.map((w, i) => ({ i, e: content(w) })).sort((a, b) => a.e - b.e).slice(0, keepN).map((o) => o.i);
    const everyone = medianOf(plainWins);
    let template = flatInside(medianOf(order.map((i) => plainWins[i])), everyone);
    order = plainWins.map((w, i) => ({ i, e: innerError(w, template) })).sort((a, b) => a.e - b.e).slice(0, keepN).map((o) => o.i);
    template = flatInside(medianOf(order.map((i) => plainWins[i])), everyone);
    return { template, shifts: exts.map(() => [0, 0]), residual: order.reduce((a, i) => a + innerError(plainWins[i], template), 0) / order.length };
  }
  // the first reference: an inner square that looks empty
  const inner = pos.map(([r, c]) => r > 0 && r < 7 && c > 0 && c < 7);
  const ref = plainWins.map((w, i) => ({ i, e: content(w) + (inner[i] ? 0 : 1) })).sort((a, b) => a.e - b.e)[0];
  let template = plainWins[ref.i];
  const cap = 0.3 * contrast;
  const shifts = exts.map(() => [0, 0]);
  let order = [], wins = plainWins;
  const rounds = quick ? 2 : 3;
  for (let it = 0; it < rounds; it++) {
    // whole pixels first; a fraction of a pixel in the last round
    exts.forEach((e, i) => { shifts[i] = bestShift(e, template, cap, pos[i][0], pos[i][1], !quick && it === rounds - 1); });
    wins = exts.map((e, i) => windowOf(e, shifts[i][0], shifts[i][1]));
    order = wins.map((w, i) => ({ i, e: innerError(w, template) })).sort((a, b) => a.e - b.e).slice(0, keepN).map((o) => o.i);
    template = medianOf(order.map((i) => wins[i]));
  }
  // how far the emptiest squares stay from the empty square after the alignment: small for hatching and regular
  // dots, large for a grain that never repeats (a grey tint dithered by a black and white scan)
  const residual = order.reduce((a, i) => a + innerError(wins[i], template), 0) / order.length;
  return { template, shifts, residual };
}

/** a window moved back by its shift (rounded to whole pixels); what comes into view is `fill` */
function unshift(d, [sx, sy], fill) {
  const dx = Math.round(sx), dy = Math.round(sy);
  if (!dx && !dy) return d;
  const out = new Float32Array(N * N).fill(fill);
  for (let y = 0; y < N; y++) {
    const yy = y - dy;
    if (yy < 0 || yy >= N) continue;
    for (let x = 0; x < N; x++) { const xx = x - dx; if (xx >= 0 && xx < N) out[y * N + x] = d[yy * N + xx]; }
  }
  return out;
}

/** 3 × 3 minimum (max = false) or maximum (max = true) of a window, edges clamped: a pass along the rows, then one
 *  along the columns (the extreme of a square is the extreme of its row extremes) */
function localExtreme(win, max) {
  const { w, h, d } = win;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let v = d[o + x];
      if (x > 0) { const u = d[o + x - 1]; if (max ? u > v : u < v) v = u; }
      if (x < w - 1) { const u = d[o + x + 1]; if (max ? u > v : u < v) v = u; }
      tmp[o + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let v = tmp[o + x];
      if (y > 0) { const u = tmp[o - w + x]; if (max ? u > v : u < v) v = u; }
      if (y < h - 1) { const u = tmp[o + w + x]; if (max ? u > v : u < v) v = u; }
      out[o + x] = v;
    }
  }
  return out;
}

/** standard deviation in the corner patches: hatching and stippling are "patterned" */
function spread(win) {
  let s = 0, s2 = 0, n = 0;
  for (const [cx, cy] of CORNERS) for (let y = cy; y < cy + CP; y++) for (let x = cx; x < cx + CP; x++) { const v = win.d[y * N + x]; s += v; s2 += v * v; n++; }
  const m = s / n;
  return Math.sqrt(Math.max(0, s2 / n - m * m));
}

/**
 * analyzeSquares(img, xs, ys, { arrows }) → { squares: [64] (row-major from the top left), lightParity, levels,
 *   patterned, arrows: [segments] }
 * square: { r, c, i, light, patch, occ (0|1), sil, ink (Uint8Array N*N), box, area, strength, interior, clean, resid, … }
 * clean: the square for the classifier (N × N, 1 = paper, 0 = ink), background and arrows white;
 * resid: the difference to the empty square (−1 … 1, positive = darker), arrows taken out
 */
export function analyzeSquares(img, xs, ys, opts = {}) {
  const exts = [];
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const mx = ((xs[c + 1] - xs[c]) * K) / N, my = ((ys[r + 1] - ys[r]) * K) / N;
    exts.push(cell(img, xs[c] - mx, ys[r] - my, xs[c + 1] + mx, ys[r + 1] + my, NE));
  }
  const cls = [[], []];
  exts.forEach((e, i) => cls[((i >> 3) + (i & 7)) % 2].push(i));
  // a first guess of the contrast (for the robust alignment): the darkest pixels against the brightest squares
  const all0 = new Float32Array(64 * N * N);
  exts.forEach((e, i) => all0.set(windowOf(e, 0, 0).d, i * N * N));
  const contrast0 = Math.max(0.25, percentile(all0, 0.9) - Math.min(0.45, percentile(all0, 0.01)));
  const aligned = cls.map((ids) => alignClass(ids.map((i) => exts[i]), ids.map((i) => [i >> 3, i & 7]), contrast0, !!opts.quick));
  const shiftOf = new Array(64);
  cls.forEach((ids, k) => ids.forEach((i, j) => { shiftOf[i] = aligned[k].shifts[j]; }));
  const patches = exts.map((e, i) => windowOf(e, ...shiftOf[i]));
  const wides = patches;
  const empty = aligned.map((a) => a.template);
  const meanOf = (p) => { let s = 0; for (let i = 0; i < N * N; i++) s += p.d[i]; return s / (N * N); };
  const lv = empty.map(meanOf);
  const lightParity = lv[0] >= lv[1] ? 0 : 1;
  const patterned = cls.map((ids) => { const v = ids.map((i) => spread(windowOf(exts[i], 0, 0))).sort((a, b) => a - b); return v[v.length >> 1] > 0.06; });
  // the ink level of the diagram: the darkest pixels of all squares; the paper: the light squares
  const all = new Float32Array(64 * N * N);
  patches.forEach((p, i) => all.set(p.d, i * N * N));
  const inkLevel = Math.min(0.45, percentile(all, 0.01));
  const paper = Math.max(lv[lightParity], percentile(empty[lightParity].d, 0.9));
  const contrast = Math.max(0.25, paper - inkLevel);
  // a grain that never repeats (a dithered grey tint) differs from its median pixel by pixel: there the evidence
  // compares local means (a white body is lighter than the grain around it, a black one darker)
  const grainy = aligned.map((al, k) => patterned[k] && al.residual > 0.08);
  // the darkest and the lightest tenth of a pattern's empty square: its lines and its gaps
  const patLevels = empty.map((t) => [percentile(t.d, 0.1), percentile(t.d, 0.9)]);
  const patMean = empty.map(meanOf), tplMean = empty.map((t, k) => (patterned[k] ? boxBlur(t, 3).d : null));
  // faint hatching: grey lines far lighter than the ink of the pieces (fine hatching printed small, a dimmed page)
  const faint = patLevels.map(([lo], k) => patterned[k] && !grainy[k] && lo - inkLevel > 0.35 * contrast);
  const wideMean = aligned.map((al, k) => (grainy[k] ? boxBlur(al.template, 2).d : null));
  // the lines and the gaps of a pattern's empty square, and their local shares, are the same for all its squares
  const rr = 4;
  const patMasks = empty.map((t, k) => {
    if (!patterned[k]) return null;
    const e = t.d, pHi = patLevels[k][1], mid = (patLevels[k][0] + pHi) / 2;
    const lines = gray(N, NH), gaps = gray(N, NH);
    for (let j = 0; j < N * NH; j++) {
      lines.d[j] = e[j] < mid && pHi - e[j] > 0.12 * contrast ? 1 : 0;
      gaps.d[j] = e[j] >= mid && e[j] - inkLevel > 0.5 * contrast ? 1 : 0;
    }
    return { lines: lines.d, gaps: gaps.d, bl: boxBlur(lines, rr).d, bg: boxBlur(gaps, rr).d };
  });
  const squares = [];
  const evid = [];
  for (let i = 0; i < 64; i++) {
    const r = i >> 3, c = i & 7, k = (r + c) % 2;
    const wv = wides[i], e = aligned[k].template.d;
    const a = { w: N, h: NH, d: new Float32Array(N * NH) };
    for (let j = 0; j < N * NH; j++) a.d[j] = Math.abs(wv.d[j] - e[j]) / contrast;
    if (patterned[k]) {
      // on hatching or stippling: the share of the pattern's dark pixels that turned light (a white body hides
      // the lines) and of its light pixels that turned dark (a black body), in a window of about a line spacing;
      // independent of how dense the pattern is
      // (a line counts as hidden only if no dark pixel is left within one pixel, a gap as filled only if no light
      // pixel is left: a pattern a pixel off is no piece)
      // a line pixel counts as hidden when most of its darkness is gone (resampling leaves thin lines grey)
      // lines and gaps of the pattern relative to its own range (faint grey hatching as well as black lines)
      const lo = localExtreme(wv, false), hi = localExtreme(wv, true);
      const hid = gray(N, NH), drk = gray(N, NH);
      const pHi = patLevels[k][1], pm = patMasks[k];
      for (let j = 0; j < N * NH; j++) {
        hid.d[j] = pm.lines[j] && lo[j] > e[j] + 0.6 * (pHi - e[j]) ? 1 : 0;
        drk.d[j] = pm.gaps[j] && hi[j] < e[j] - 0.65 * (e[j] - inkLevel) ? 1 : 0;
      }
      const bh = boxBlur(hid, rr).d, bd = boxBlur(drk, rr).d, bl = pm.bl, bg = pm.bg;
      let sharp;
      if (grainy[k]) {
        const mw = boxBlur(wv, 2).d, me = wideMean[k];
        sharp = new Float32Array(N * NH);
        for (let j = 0; j < N * NH; j++) sharp[j] = Math.abs(mw[j] - me[j]) / contrast;
      } else sharp = blur(a, 0.7).d;
      // local means: a white body is lighter than the pattern around it, however fine and grey the hatching is
      // (its gaps may be grey themselves), a black body darker; measured against the pattern's own depth
      const mw = boxBlur(wv, 3).d, me = tplMean[k], depth = Math.max(0.06, paper - patMean[k]);
      for (let j = 0; j < N * NH; j++) {
        const share = Math.max(bl[j] > 0.04 ? bh[j] / bl[j] : 0, bg[j] > 0.04 ? bd[j] / bg[j] : 0);
        const light = (mw[j] - me[j]) / depth, dark = (me[j] - mw[j]) / contrast;
        a.d[j] = Math.max(sharp[j], share * 0.9, faint[k] ? 0.8 * Math.max(light, dark) : 0);
      }
    }
    // what the shift brought into view from beyond the square belongs to the neighbour
    const [sx, sy] = shiftOf[i];
    if (sx || sy) {
      const xa = Math.max(0, -Math.floor(sx)), xb = Math.min(N, N - Math.ceil(sx)), ya = Math.max(0, -Math.floor(sy)), yb = Math.min(NH, NH - Math.ceil(sy));
      for (let y = 0; y < NH; y++) for (let x = 0; x < N; x++) if (y < ya || y >= yb || x < xa || x >= xb) a.d[y * N + x] = 0;
    }
    // thin outlines of white pieces keep their full strength
    evid.push(a);
    squares.push({ r, c, i, light: k === lightParity, parity: k, patch: patches[i], shift: shiftOf[i] });
  }
  // noise level per class: the evidence near the square corners (pieces rarely reach them)
  const noise = [0, 1].map((k) => {
    const v = [];
    squares.filter((s) => s.parity === k).forEach((s) => { const a = evid[s.i].d; for (const [x, y] of [[7, 7], [N - 8, 7], [7, N - 8], [N - 8, N - 8]]) v.push(a[y * N + x]); });
    return percentile(v, 0.6);
  });
  for (const s of squares) s.t = Math.max(patterned[s.parity] ? 0.3 : 0.2, 3 * noise[s.parity]);
  // arrows and lines over the board: found across all squares, then no evidence of a piece
  let arrows = null;
  if (opts.arrows !== false) {
    arrows = findArrows(evid.map((a, i) => { const t = squares[i].t, out = new Float32Array(N * N); for (let j = 0; j < N * N; j++) out[j] = a.d[j] / t; return out; }), squares, N, M, opts.debug || null);
    for (const [i, m] of arrows.masks) {
      squares[i].arrow = m;
      const a = evid[i].d;
      for (let j = 0; j < N * N; j++) if (m[j]) a[j] = 0;
    }
  }
  for (const s of squares) {
    const a = evid[s.i].d, k = s.parity;
    const t = s.t;
    // the window without the board's outer edge (a frame, also where the shift brings it into view) and without the
    // square's outermost pixel column/row
    const iw = inside(s.r, s.c, s.shift[0], s.shift[1]);
    const y0w = Math.max(1, iw.ya), y1w = Math.min(NH, iw.yb), x0w = Math.max(1, iw.xa), x1w = Math.min(N - 1, iw.xb);
    // the square for the classifier: what matches the empty square (paper, hatching) and the arrows become paper,
    // the rest keeps its printed grey; a pixel matches when most of its neighbourhood does (inside a black body a
    // single pixel on a dark hatch line stays dark)
    const e = empty[k].d, p = s.patch.d;
    const match = gray(N, N);
    for (let j = 0; j < N * N; j++) match.d[j] = Math.abs(p[j] - e[j]) < 0.18 * contrast || (s.arrow && s.arrow[j]) ? 1 : 0;
    const share = boxBlur(match, 2).d;
    const clean = new Float32Array(N * N), resid = new Float32Array(N * N);
    for (let j = 0; j < N * N; j++) {
      const v = Math.max(0, Math.min(1, (p[j] - inkLevel) / contrast));
      const arrow = s.arrow && s.arrow[j];
      clean[j] = arrow || (match.d[j] && share[j] > 0.6) ? 1 : v;
      resid[j] = arrow ? 0 : Math.max(-1, Math.min(1, (e[j] - p[j]) / contrast));
    }
    // back where the square lies on the board (the window was shifted onto the pattern): the classifier sees every
    // piece in its place in the square
    s.clean = unshift(clean, s.shift, 1); s.resid = unshift(resid, s.shift, 0);
    const raw = new Uint8Array(N * NH);
    // on a pattern the outline of a white piece is weak evidence (thin grey hatching is about as dark as a thin
    // outline at a small size): the dark pixels of the cleaned square, where the pattern is gone, count as well
    const darkClean = faint[k] ? 0.5 : -1;
    let rawN = 0;
    for (let y = y0w; y < y1w; y++) for (let x = x0w; x < x1w; x++) { const j = y * N + x; if (a[j] > t || clean[j] < darkClean) { raw[j] = 1; rawN++; } }
    s.rawShare = rawN / (N * N);
    const close = (rad) => {
      const m = erode(dilate(raw, N, NH, rad), N, NH, rad);
      // a piece standing on the bottom of its square may lose the bottom stroke of its outline to the square's
      // edge (the grid a pixel off): close the outline along the bottom before filling it
      const yb = y1w - 1;
      let bx0 = N, bx1 = -1;
      for (let y = yb - 2; y <= yb; y++) for (let x = x0w; x < x1w; x++) if (m[y * N + x]) { if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; }
      if (bx1 - bx0 >= N * 0.25 && bx1 - bx0 <= N * 0.85) for (let x = bx0; x <= bx1; x++) m[yb * N + x] = 1;
      return fillHoles(m, N, NH);
    };
    let m = close(2);
    // the outline of a white piece in a black and white scan breaks up into dots, and a line drawn across a piece
    // leaves gaps in its outline: when the narrow closing leaves little more than a thin outline, a wider one closes it
    if (!patterned[k] || s.arrow) {
      const fill = (mm) => {
        let nr = 0, nf = 0, x0r = N, x1r = -1, y0r = NH, y1r = -1;
        for (let y = Math.round(N * 0.08); y < NH; y++) for (let x = Math.round(N * 0.15); x < Math.round(N * 0.85); x++) {
          const j = y * N + x;
          if (mm[j]) nf++;
          if (raw[j]) { nr++; if (x < x0r) x0r = x; if (x > x1r) x1r = x; if (y < y0r) y0r = y; if (y > y1r) y1r = y; }
        }
        return { nr, nf, h: y1r - y0r + 1, density: nr / Math.max(1, (x1r - x0r + 1) * (y1r - y0r + 1)) };
      };
      const f2 = fill(m);
      if (f2.h > N * 0.3 && f2.density < 0.35 && f2.nf < 1.6 * f2.nr) {
        for (const rad of [3, 4]) {
          const mm = close(rad), f = fill(mm);
          if (f.nf >= 2 * f.nr) { m = mm; break; }
        }
      }
    }
    // keep the components with a good part of their mass in the middle of the square (a neighbour's piece or a
    // frame may reach into the window from the side)
    const { labels, comps } = components(m, N, NH);
    const inMid = new Int32Array(comps.length + 1);
    const lo = Math.round(N * 0.2), hi = Math.round(N * 0.8);
    for (let y = lo; y < hi; y++) for (let x = lo; x < hi; x++) inMid[labels[y * N + x]]++;
    const keep = new Set();
    comps.forEach((cp, j) => { if (cp.n >= N * N * 0.012 && inMid[j + 1] >= Math.max(8, 0.3 * cp.n)) keep.add(j + 1); });
    // back to the square
    const sil = new Uint8Array(N * N);
    let area = 0, x0 = N, y0 = N, x1 = -1, y1 = -1, strong = 0;
    const tStrong = Math.max(0.4, 3 * noise[k]);
    const evs = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const jw = y * N + x;
      if (!keep.has(labels[jw])) continue;
      sil[y * N + x] = 1; area++; evs.push(a[jw]); if (a[jw] > tStrong || clean[jw] < darkClean) strong++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    s.sil = sil; s.area = area / (N * N);
    s.box = area ? { x0, y0, x1, y1, w: x1 - x0 + 1, h: y1 - y0 + 1 } : null;
    // strength: the strong part of the evidence against the threshold, or enough clearly differing pixels (a white
    // piece on white differs only along its outline, one on hatching only where it hides lines)
    s.strength = area ? Math.max(percentile(evs, 0.85) / t, (strong / (N * N)) / 0.018) : 0;
    s.evidence = a;
    // ink: dark pixels inside the silhouette (the outline and the body of black pieces)
    const inkT = inkLevel + 0.5 * Math.max(0.15, Math.min(lv[k], paper) - inkLevel);
    const ink = new Uint8Array(N * N);
    let inkN = 0;
    for (let j = 0; j < N * N; j++) if (sil[j] && s.patch.d[j] < inkT && !(s.arrow && s.arrow[j])) { ink[j] = 1; inkN++; }
    s.ink = ink; s.inkShare = area ? inkN / area : 0;
    // interior darkness: the share of ink in the silhouette shrunk by a seventh of its size; ring darkness: the
    // share of ink just inside the outline (a white piece is white there whatever its inner lines, a black piece
    // is black)
    if (s.box) {
      const er = Math.max(1, Math.round(Math.min(s.box.w, s.box.h) / 7));
      const inner = erode(sil, N, N, er);
      const r1 = erode(sil, N, N, 2), r2 = erode(sil, N, N, 5);
      let n = 0, d = 0, lum = 0, rn = 0, rd = 0;
      for (let j = 0; j < N * N; j++) {
        if (inner[j]) { n++; if (ink[j]) d++; lum += s.patch.d[j]; }
        if (r1[j] && !r2[j]) { rn++; if (ink[j]) rd++; }
      }
      s.interior = n >= 8 ? d / n : s.inkShare;
      s.interiorLum = n >= 8 ? lum / n : 0.5;
      s.ring = rn >= 8 ? rd / rn : s.interior;
    } else { s.interior = 0; s.interiorLum = 1; s.ring = 0; }
    s.darkness = 0.5 * s.interior + 0.5 * s.ring;
    s.occ = s.box && s.area > 0.035 && s.box.h > N * 0.22 && s.strength > 1.2 ? 1 : 0;
  }
  return {
    squares, lightParity, empty,
    levels: { light: lv[lightParity], dark: lv[1 - lightParity], ink: inkLevel, paper, noise },
    patterned: { light: patterned[lightParity], dark: patterned[1 - lightParity] },
    residual: { light: aligned[lightParity].residual, dark: aligned[1 - lightParity].residual },
    arrows: arrows ? arrows.segments : [], arrowBoard: arrows?.board || null,
  };
}
