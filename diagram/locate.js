// Board localisation in a snip: skew, then the 8 × 8 grid. Book diagrams differ a lot (hatched, grey, stippled or
// coloured dark squares, thick, thin, double or no frame, coordinates, captions and text around), so the grid is
// searched in two ways and the candidates compete:
//   lattices   the square boundaries form 9 equally spaced lines in both directions, where light and dark squares
//              meet; they show as peaks in projection profiles of the edge strength of a blurred copy (the blur
//              turns hatching into a flat grey), of a sharper copy (thin grid lines) and of a texture map
//              (stippling, light hatching)
//   frames     long dark lines around the board (a black and white scan may have lost the grey squares)
// Each candidate is scored on how completely its inner lines are square boundaries along their whole length, on
// the checkerboard of square backgrounds, on a frame around it and on its size in the snip; the winner is refined
// on the profiles inside the board.
import { resize, blur, rotate, median, percentile, histQuantile, gray } from './image.js';

/** gradient magnitudes |∂x| and |∂y| (central differences) */
function gradients(img) {
  const { w, h, d } = img;
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) gx[row + x] = Math.abs(d[row + x + 1] - d[row + x - 1]);
  }
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) gy[row + x] = Math.abs(d[row + w + x] - d[row - w + x]);
  }
  return { gx, gy, w, h };
}

/** prefix sums along columns of gx and along rows of gy: segment sums of a vertical/horizontal line in O(1) */
function linePrefix(g) {
  const { gx, gy, w, h } = g;
  const cx = new Float32Array(w * (h + 1)), ry = new Float32Array(h * (w + 1));
  for (let x = 0; x < w; x++) { let s = 0; for (let y = 0; y < h; y++) { s += gx[y * w + x]; cx[x * (h + 1) + y + 1] = s; } }
  for (let y = 0; y < h; y++) { let s = 0; for (let x = 0; x < w; x++) { s += gy[y * w + x]; ry[y * (w + 1) + x + 1] = s; } }
  return {
    w, h,
    /** sum of gx over the column x (rounded) between y0 and y1 */
    col(x, y0, y1) { x = Math.round(x); if (x < 0 || x >= w) return 0; y0 = Math.max(0, Math.min(h, Math.round(y0))); y1 = Math.max(y0, Math.min(h, Math.round(y1))); return cx[x * (h + 1) + y1] - cx[x * (h + 1) + y0]; },
    row(y, x0, x1) { y = Math.round(y); if (y < 0 || y >= h) return 0; x0 = Math.max(0, Math.min(w, Math.round(x0))); x1 = Math.max(x0, Math.min(w, Math.round(x1))); return ry[y * (w + 1) + x1] - ry[y * (w + 1) + x0]; },
  };
}

/**
 * Skew in degrees such that rotate(img, skew) makes the board lines axis-parallel (search −3°…3°). The strongest
 * edge pixels are projected onto the rotated axes; the right angle concentrates them in the fewest bins.
 */
export function estimateSkew(img) {
  const sc = Math.min(1, 420 / Math.max(img.w, img.h));
  const s = sc < 1 ? resize(img, img.w * sc, img.h * sc) : img;
  const { w, h } = s;
  const { gx, gy } = gradients(blur(s, 0.7));
  const pick = (g) => {
    const t = Math.max(0.08, histQuantile(g, 0.88));
    const pts = [];
    for (let i = 0; i < g.length; i++) if (g[i] >= t) pts.push(i);
    return pts;
  };
  const hx = pick(gx), hy = pick(gy);
  if (hx.length + hy.length < 50) return 0;
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  const diag = Math.ceil(Math.hypot(w, h)) + 4;
  const score = (deg) => {
    const a = (deg * Math.PI) / 180, c = Math.cos(a), sn = Math.sin(a);
    const hb = new Float32Array(diag * 2), vb = new Float32Array(diag * 2);
    for (const i of hy) { const x = i % w, y = (i - x) / w; hb[Math.round(sn * (x - cx) + c * (y - cy)) + diag] += gy[i]; }
    for (const i of hx) { const x = i % w, y = (i - x) / w; vb[Math.round(c * (x - cx) - sn * (y - cy)) + diag] += gx[i]; }
    let e = 0;
    for (let k = 0; k < hb.length; k++) e += hb[k] * hb[k] + vb[k] * vb[k];
    return e;
  };
  let best = 0, bestScore = -1;
  const base = score(0);
  for (let deg = -3; deg <= 3.0001; deg += 0.25) { const v = score(deg); if (v > bestScore) { bestScore = v; best = deg; } }
  for (let deg = best - 0.2; deg <= best + 0.2001; deg += 0.05) { const v = score(deg); if (v > bestScore) { bestScore = v; best = deg; } }
  return bestScore > base * 1.03 ? Math.round(best * 100) / 100 : 0;
}

/** high-pass profile: subtract a moving average, keep the positive part, normalise to max 1 */
function peaks(profile, win) {
  const n = profile.length, out = new Float32Array(n);
  const half = Math.max(2, Math.round(win / 2));
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + profile[i];
  let max = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half), b = Math.min(n, i + half + 1);
    const v = profile[i] - (pre[b] - pre[a]) / (b - a);
    out[i] = v > 0 ? v : 0;
    if (out[i] > max) max = out[i];
  }
  if (max > 0) for (let i = 0; i < n; i++) out[i] /= max;
  return out;
}

/** running maximum within ±r */
function maxFilter(p, r) {
  const n = p.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) { let m = 0; for (let t = Math.max(0, i - r); t <= Math.min(n - 1, i + r); t++) if (p[t] > m) m = p[t]; out[i] = m; }
  return out;
}
const at = (p, x) => { const i = Math.round(x); return i >= 0 && i < p.length ? p[i] : 0; };

/** best lattices x0 + k·p (k = 0…8) in a profile; for each period the best offsets that are a square apart */
function lattices(prof, n, pMin, pMax, keep) {
  const dil = [1, 2, 3].map((r) => maxFilter(prof, r));
  const cands = [];
  const xsBuf = new Float32Array(Math.ceil(2 * n) + 8), ssBuf = new Float32Array(Math.ceil(2 * n) + 8);
  for (let p = pMin; p <= pMax; p += Math.max(0.25, p / 160)) {
    const pd = dil[Math.min(2, Math.max(0, Math.round(p / 40) - 1))];
    const step = Math.max(0.5, p / 60);
    let m = 0;
    for (let x0 = -p * 0.08; x0 + 8 * p <= n - 1 + p * 0.08; x0 += step) {
      let s = 0;
      for (let k = 0; k <= 8; k++) { const i = Math.round(x0 + k * p); if (i >= 0 && i < n) s += pd[i]; }
      xsBuf[m] = x0; ssBuf[m++] = s;
    }
    // the three best offsets at least half a square apart (greedy non-maximum suppression)
    for (let t = 0; t < 3; t++) {
      let bi = -1, bs = 0;
      for (let i = 0; i < m; i++) if (ssBuf[i] > bs) { bs = ssBuf[i]; bi = i; }
      if (bi < 0) break;
      const bx = xsBuf[bi];
      cands.push({ x0: bx, p, s: bs / 9 });
      for (let i = 0; i < m; i++) if (Math.abs(xsBuf[i] - bx) < p * 0.5) ssBuf[i] = 0;
    }
  }
  cands.sort((a, b) => b.s - a.s);
  const out = [];
  for (const c of cands) {
    if (out.some((o) => Math.abs(o.p - c.p) < o.p * 0.03 && Math.abs(o.x0 - c.x0) < o.p * 0.25)) continue;
    out.push(c);
    if (out.length >= keep) break;
  }
  return out;
}

/** long dark lines (frames): bands of rows/columns whose longest dark run is long; boxes made of four of them */
function frameLattices(b) {
  const { w, h } = b;
  const dark = Math.min(0.62, (percentile(b.d, 0.01) + percentile(b.d, 0.7)) / 2);
  const maxGap = Math.max(1, Math.round(Math.min(w, h) / 160));   // thin lines of a black and white scan break up
  const runs = (len, other, val) => {
    const out = [];
    for (let i = 0; i < len; i++) {
      let run = 0, best = 0, start = 0, bestStart = 0, gap = 0;
      for (let j = 0; j < other; j++) {
        if (val(i, j) < dark) { if (!run) start = j; run += 1 + gap; gap = 0; if (run > best) { best = run; bestStart = start; } }
        else if (run && gap < maxGap) gap++;
        else { run = 0; gap = 0; }
      }
      out.push({ i, len: best, a: bestStart, b: bestStart + best });
    }
    return out;
  };
  const rows = runs(h, w, (y, x) => b.d[y * w + x]).filter((r) => r.len >= 0.25 * w);
  const cols = runs(w, h, (x, y) => b.d[y * w + x]).filter((r) => r.len >= 0.25 * h);
  const bands = (list) => {
    const out = [];
    for (const r of list) {
      const last = out[out.length - 1];
      if (last && r.i <= last.i1 + 1) { last.i1 = r.i; last.a = Math.min(last.a, r.a); last.b = Math.max(last.b, r.b); }
      else out.push({ i0: r.i, i1: r.i, a: r.a, b: r.b });
    }
    return out;
  };
  const hb = bands(rows).slice(0, 30), vb = bands(cols).slice(0, 30);
  const out = [];
  for (let t = 0; t < hb.length; t++) for (let bo = t + 1; bo < hb.length; bo++) {
    const top = hb[t], bot = hb[bo];
    const H = bot.i0 - top.i1 - 1;
    if (H < 0.2 * Math.min(w, h)) continue;
    for (let l = 0; l < vb.length; l++) for (let r = l + 1; r < vb.length; r++) {
      const left = vb[l], right = vb[r];
      const W = right.i0 - left.i1 - 1;
      if (Math.abs(W - H) > 0.05 * Math.max(W, H)) continue;
      // the lines must span the box
      if (top.a > left.i1 + 3 || top.b < right.i0 - 3 || bot.a > left.i1 + 3 || bot.b < right.i0 - 3) continue;
      if (left.a > top.i1 + 3 || left.b < bot.i0 - 3 || right.a > top.i1 + 3 || right.b < bot.i0 - 3) continue;
      out.push({ x0: left.i1 + 1, p: W / 8, y0: top.i1 + 1, q: H / 8, s: 1 });
    }
  }
  return out.slice(0, 16);
}

/** per-cell background estimate: the median of the four corner patches of each square */
function cellBackgrounds(img, xs, ys) {
  const b = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const x0 = xs[c], x1 = xs[c + 1], y0 = ys[r], y1 = ys[r + 1];
      const cw = x1 - x0, ch = y1 - y0;
      const vals = [];
      for (const [fx, fy] of [[0.1, 0.1], [0.75, 0.1], [0.1, 0.75], [0.75, 0.75]]) {
        for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
          const x = Math.round(x0 + cw * (fx + i * 0.06)), y = Math.round(y0 + ch * (fy + j * 0.06));
          vals.push(x >= 0 && y >= 0 && x < img.w && y < img.h ? img.d[y * img.w + x] : 1);
        }
      }
      b.push(median(vals));
    }
  }
  return b;
}

/** how clearly the cells alternate (light/dark parity): mean difference over the spread inside the classes */
function checkerScore(bgs) {
  const a = [], b = [];
  for (let i = 0; i < 64; i++) (((i >> 3) + (i & 7)) % 2 ? b : a).push(bgs[i]);
  const ma = a.reduce((s, v) => s + v, 0) / 32, mb = b.reduce((s, v) => s + v, 0) / 32;
  const sd = (arr, m) => Math.sqrt(arr.reduce((s, v) => s + (v - m) * (v - m), 0) / arr.length);
  return { contrast: Math.abs(ma - mb), z: Math.abs(ma - mb) / (0.5 * (sd(a, ma) + sd(b, mb)) + 0.012), lightParity: ma >= mb ? 0 : 1 };
}

/** local texture energy: blurred gradient magnitude of a lightly blurred copy (stippling, hatching) */
function texture(img, sigma) {
  const { gx, gy } = gradients(blur(img, 0.6));
  const t = gray(img.w, img.h);
  for (let i = 0; i < t.d.length; i++) t.d[i] = Math.min(1, (gx[i] + gy[i]) * 2);
  return blur(t, sigma);
}

/**
 * Square boundary coverage: for each inner line, the share of its 8 segments (square sides) that carry a clear
 * edge. A checkerboard has an edge on every square side; lines that merely pass along pieces have a few.
 */
function coverage(P, xs, ys) {
  const r = Math.max(1, Math.round((xs[8] - xs[0]) / 8 / 24));
  // reference per direction: edge strength through the square centres (pieces, hatching), measured like the
  // lines themselves (maximum over ±r), robust median
  const refX = [], refY = [];
  const segX = (x, s, o) => { const len = Math.max(1, ys[s + 1] - ys[s]); return P.col(x + o, ys[s] + 0.15 * len, ys[s + 1] - 0.15 * len) / (0.7 * len); };
  const segY = (y, s, o) => { const len = Math.max(1, xs[s + 1] - xs[s]); return P.row(y + o, xs[s] + 0.15 * len, xs[s + 1] - 0.15 * len) / (0.7 * len); };
  for (let c = 0; c < 8; c++) for (let s = 0; s < 8; s++) {
    let mx = 0, my = 0;
    for (let o = -r; o <= r; o++) { mx = Math.max(mx, segX((xs[c] + xs[c + 1]) / 2, s, o)); my = Math.max(my, segY((ys[c] + ys[c + 1]) / 2, s, o)); }
    refX.push(mx); refY.push(my);
  }
  const baseX = Math.max(0.004, median(refX)), baseY = Math.max(0.004, median(refY));
  let hits = 0, n = 0;
  for (let k = 1; k < 8; k++) {
    for (let s = 0; s < 8; s++) {
      let mx = 0, my = 0;
      for (let o = -r; o <= r; o++) { mx = Math.max(mx, segX(xs[k], s, o)); my = Math.max(my, segY(ys[k], s, o)); }
      hits += (mx > 2.5 * baseX && mx > 0.02 ? 1 : 0) + (my > 2.5 * baseY && my > 0.02 ? 1 : 0);
      n += 2;
    }
  }
  return hits / n;
}

/**
 * Square boundaries in the bands along the edges of a lattice: for the outermost row and column on each side and for
 * the band one square beyond it, the share of the seven inner lines that carry an edge across that band (measured
 * like coverage(), on several edge maps, the strongest counts). A lattice one square off puts a row of text
 * (coordinates, a caption) inside the board, where no line crosses, and leaves a board row outside, where all do.
 */
function edgeBands(Ps, xs, ys) {
  const p = (xs[8] - xs[0]) / 8, q = (ys[8] - ys[0]) / 8;
  const r = Math.max(1, Math.round(p / 24));
  const tests = Ps.map((P) => {
    const segX = (x, a, b) => { const len = Math.max(1, b - a); return P.col(x, a + 0.15 * len, b - 0.15 * len) / (0.7 * len); };
    const segY = (y, a, b) => { const len = Math.max(1, b - a); return P.row(y, a + 0.15 * len, b - 0.15 * len) / (0.7 * len); };
    const refX = [], refY = [];
    for (let c = 0; c < 8; c++) for (let s = 0; s < 8; s++) {
      let mx = 0, my = 0;
      for (let o = -r; o <= r; o++) { mx = Math.max(mx, segX((xs[c] + xs[c + 1]) / 2 + o, ys[s], ys[s + 1])); my = Math.max(my, segY((ys[c] + ys[c + 1]) / 2 + o, xs[s], xs[s + 1])); }
      refX.push(mx); refY.push(my);
    }
    const baseX = Math.max(0.004, median(refX)), baseY = Math.max(0.004, median(refY));
    const hit = (seg, base, line, a, b) => { let m = 0; for (let o = -r; o <= r; o++) m = Math.max(m, seg(line + o, a, b)); return m > 2.5 * base && m > 0.02; };
    // the inner vertical lines across the row band [a, b], the inner horizontal ones across the column band
    return {
      row: (a, b) => { let n = 0; for (let k = 1; k < 8; k++) if (hit(segX, baseX, xs[k], a, b)) n++; return n / 7; },
      col: (a, b) => { let n = 0; for (let k = 1; k < 8; k++) if (hit(segY, baseY, ys[k], a, b)) n++; return n / 7; },
    };
  });
  const row = (a, b) => Math.max(...tests.map((t) => t.row(a, b))), col = (a, b) => Math.max(...tests.map((t) => t.col(a, b)));
  return {
    above: row(ys[0] - q, ys[0]), top: row(ys[0], ys[1]), bottom: row(ys[7], ys[8]), below: row(ys[8], ys[8] + q),
    beforeX: col(xs[0] - p, xs[0]), first: col(xs[0], xs[1]), last: col(xs[7], xs[8]), afterX: col(xs[8], xs[8] + p),
  };
}

/** dark line evidence along the four sides of a box (frames), 0…1 */
function frameEvidence(img, x0, y0, x1, y1) {
  const { w, h, d } = img;
  const off = Math.max(2, (x1 - x0) / 45);
  const side = (ax, ay, bx, by, vertical) => {
    let best = 0;
    const n = 32;
    for (let o = -off; o <= off; o += 1) {
      let dark = 0;
      for (let k = 0; k < n; k++) {
        const t = (k + 0.5) / n;
        const x = Math.round(ax + (bx - ax) * t + (vertical ? o : 0)), y = Math.round(ay + (by - ay) * t + (vertical ? 0 : o));
        if (x >= 0 && y >= 0 && x < w && y < h && d[y * w + x] < 0.5) dark++;
      }
      if (dark / n > best) best = dark / n;
    }
    return best;
  };
  return (side(x0, y0, x1, y0, false) + side(x0, y1, x1, y1, false) + side(x0, y0, x0, y1, true) + side(x1, y0, x1, y1, true)) / 4;
}

/** refine 9 lattice lines on a profile restricted to the board; returns a straight lattice fitted to them */
function refine(prof, x0, p) {
  const pos = [];
  let mean = 0; for (const v of prof) mean += v; mean /= prof.length || 1;
  for (let k = 0; k <= 8; k++) {
    const c = x0 + k * p, r = p * 0.16;
    let best = -1, bx = c;
    for (let t = Math.max(0, Math.floor(c - r)); t <= Math.min(prof.length - 1, Math.ceil(c + r)); t++) if (prof[t] > best) { best = prof[t]; bx = t; }
    if (bx > 0 && bx < prof.length - 1) {
      const a = prof[bx - 1], m = prof[bx], b = prof[bx + 1], den = a - 2 * m + b;
      if (den < 0) bx += Math.max(-0.5, Math.min(0.5, (0.5 * (a - b)) / den));
    }
    pos.push({ k, x: bx, w: best });
  }
  // only clear inner lines (square boundaries) count; the outer ones may sit on a frame
  let pts = pos.filter((q) => q.k >= 1 && q.k <= 7 && q.w > mean * 1.8);
  const fit = (arr) => {
    const n = arr.length, sk = arr.reduce((s, q) => s + q.k, 0), sx = arr.reduce((s, q) => s + q.x, 0);
    const skk = arr.reduce((s, q) => s + q.k * q.k, 0), skx = arr.reduce((s, q) => s + q.k * q.x, 0);
    const den = n * skk - sk * sk;
    const b = den ? (n * skx - sk * sx) / den : p;
    return { a: (sx - b * sk) / n, b };
  };
  if (pts.length < 4) return { lines: Array.from({ length: 9 }, (_, k) => x0 + k * p), p, refined: false };
  let f = fit(pts);
  for (let it = 0; it < 2; it++) {
    const res = pts.map((q) => Math.abs(q.x - (f.a + f.b * q.k)));
    const lim = Math.max(1, p * 0.06);
    const kept = pts.filter((_, i) => res[i] <= lim);
    if (kept.length >= 4 && kept.length < pts.length) { pts = kept; f = fit(pts); }
  }
  if (Math.abs(f.b - p) > p * 0.08) return { lines: Array.from({ length: 9 }, (_, k) => x0 + k * p), p, refined: false };
  return { lines: Array.from({ length: 9 }, (_, k) => f.a + f.b * k), p: f.b, refined: true };
}

/**
 * Finds the board in a grey snip. Returns null without any candidate, else
 * { xs: [9], ys: [9] (line positions in pixels of `img`), ok, score, checker, skew, img (the deskewed snip) }.
 */
export function findBoard(src, opts = {}) {
  let img = src;
  const skew = opts.skew ?? estimateSkew(src);
  if (Math.abs(skew) >= 0.1) img = rotate(src, skew);
  const sc = Math.min(1, 760 / Math.max(img.w, img.h));
  const work = sc < 1 ? resize(img, img.w * sc, img.h * sc) : img;
  const { w, h } = work;
  const minSide = Math.min(w, h);
  const pMin = Math.max(6, (minSide * 0.2) / 8), pMax = minSide / 7.95;
  if (pMax < pMin) return null;
  const sigmaC = Math.max(1.2, minSide / 130);
  const coarse = blur(work, sigmaC), fineImg = blur(work, 0.8);
  const tex = texture(work, sigmaC * 1.3);
  const G = { coarse: gradients(coarse), fine: gradients(fineImg), tex: gradients(tex) };
  const cands = [];
  for (const [tag, g] of Object.entries(G)) {
    const px = new Float32Array(w), py = new Float32Array(h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { px[x] += g.gx[y * w + x]; py[y] += g.gy[y * w + x]; }
    const lx = lattices(peaks(px, minSide / 14), w, pMin, pMax, 6), ly = lattices(peaks(py, minSide / 14), h, pMin, pMax, 6);
    for (const cx of lx) for (const cy of ly) {
      if (Math.abs(cx.p - cy.p) > 0.12 * Math.max(cx.p, cy.p)) continue;
      cands.push({ x0: cx.x0, p: cx.p, y0: cy.x0, q: cy.p, tag });
    }
  }
  for (const f of frameLattices(work)) cands.push({ ...f, tag: 'frame' });
  if (!cands.length) return null;
  const P = { coarse: linePrefix(G.coarse), fine: linePrefix(G.fine), tex: linePrefix(G.tex) };
  // profiles inside a box (fine edges plus the coarse ones for hatching and the texture ones for stippling)
  const boxProfile = (dir, a0, a1) => {
    const n = dir === 'x' ? w : h;
    // the coarse blur leads: in the fine one, hatch lines inside the squares are stronger than the boundaries
    const parts = [[P.fine, 0.3], [P.coarse, 1], [P.tex, 0.5]].map(([pp, wgt]) => {
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = dir === 'x' ? pp.col(i, a0, a1) : pp.row(i, a0, a1);
      let m = 0; for (const v of out) if (v > m) m = v;
      if (m) for (let i = 0; i < n; i++) out[i] *= wgt / m;
      return out;
    });
    return parts[0].map((v, i) => v + parts[1][i] + parts[2][i]);
  };
  const lattice = (x0, p, y0, q) => ({ xs: Array.from({ length: 9 }, (_, k) => x0 + k * p), ys: Array.from({ length: 9 }, (_, k) => y0 + k * q) });
  const inside = ({ xs, ys }) => xs[0] >= -(xs[1] - xs[0]) * 0.15 && ys[0] >= -(ys[1] - ys[0]) * 0.15 && xs[8] <= w - 1 + (xs[1] - xs[0]) * 0.15 && ys[8] <= h - 1 + (ys[1] - ys[0]) * 0.15;
  // cheap pre-score: checkerboard of the coarse and texture maps
  const pre = [];
  const seen = [];
  for (const cand of cands) {
    if (seen.some((o) => Math.abs(o.x0 - cand.x0) < cand.p * 0.08 && Math.abs(o.y0 - cand.y0) < cand.q * 0.08 && Math.abs(o.p - cand.p) < cand.p * 0.025 && Math.abs(o.q - cand.q) < cand.q * 0.025)) continue;
    seen.push(cand);
    const L = lattice(cand.x0, cand.p, cand.y0, cand.q);
    if (!inside(L)) continue;
    const ch = checkerScore(cellBackgrounds(coarse, L.xs, L.ys)), tch = checkerScore(cellBackgrounds(tex, L.xs, L.ys));
    pre.push({ cand, s: Math.min(ch.z, 6) + 8 * Math.min(ch.contrast, 0.4) + 0.8 * Math.min(tch.z, 6) + (cand.tag === 'frame' ? 3 : 0) });
  }
  pre.sort((a, b) => b.s - a.s);
  // the best candidates are refined on their own profiles, then scored in full
  let best = null;
  const done = [];
  for (const { cand } of pre.slice(0, 14)) {
    const L0 = lattice(cand.x0, cand.p, cand.y0, cand.q);
    const rx = refine(boxProfile('x', L0.ys[0], L0.ys[8]), cand.x0, cand.p);
    const ry = refine(boxProfile('y', L0.xs[0], L0.xs[8]), cand.y0, cand.q);
    const xs = rx.lines, ys = ry.lines;
    if (!inside({ xs, ys })) continue;
    if (done.some((o) => Math.abs(o.xs[0] - xs[0]) < 1.5 && Math.abs(o.ys[0] - ys[0]) < 1.5 && Math.abs(o.xs[8] - xs[8]) < 1.5 && Math.abs(o.ys[8] - ys[8]) < 1.5)) continue;
    const ch = checkerScore(cellBackgrounds(coarse, xs, ys));
    const tch = checkerScore(cellBackgrounds(tex, xs, ys));
    const cov = Math.max(coverage(P.coarse, xs, ys), coverage(P.fine, xs, ys), coverage(P.tex, xs, ys));
    const frame = frameEvidence(fineImg, xs[0], ys[0], xs[8], ys[8]);
    const size = (xs[8] - xs[0] + ys[8] - ys[0]) / 2 / minSide;
    const score = 6 * cov + 1.2 * Math.min(ch.z, 8) + 10 * Math.min(ch.contrast, 0.4) + 0.8 * Math.min(tch.z, 8) + 4 * frame + 2 * size;
    const e = { score, cand, xs, ys, checker: ch, tchecker: tch, cov, frame, size, refined: rx.refined && ry.refined };
    done.push(e);
    if (opts.debug) opts.debug.push({ tag: cand.tag, x0: +xs[0].toFixed(1), p: +((xs[8] - xs[0]) / 8).toFixed(2), y0: +ys[0].toFixed(1), q: +((ys[8] - ys[0]) / 8).toFixed(2), score: +score.toFixed(2), cov: +cov.toFixed(2), z: +ch.z.toFixed(2), con: +ch.contrast.toFixed(3), tz: +tch.z.toFixed(2), frame: +frame.toFixed(2), size: +size.toFixed(2), sc });
    if (!best || score > best.score) best = e;
  }
  if (!best) return null;
  // a lattice one square off along an axis still has 8 of its 9 lines on boundaries: the checkerboard and the
  // frame decide between it and its neighbours
  const shiftScore = (xs, ys) => {
    const ch = checkerScore(cellBackgrounds(coarse, xs, ys)), tch = checkerScore(cellBackgrounds(tex, xs, ys));
    return 1.2 * Math.min(ch.z, 8) + 10 * Math.min(ch.contrast, 0.4) + 0.8 * Math.min(tch.z, 8) + 4 * frameEvidence(fineImg, xs[0], ys[0], xs[8], ys[8]);
  };
  const p0 = (best.xs[8] - best.xs[0]) / 8, q0 = (best.ys[8] - best.ys[0]) / 8;
  let shift = { dx: 0, dy: 0, s: shiftScore(best.xs, best.ys) + 0.3 };
  for (const dx of [-1, 0, 1]) for (const dy of [-1, 0, 1]) {
    if (!dx && !dy) continue;
    const xs = best.xs.map((v) => v + dx * p0), ys = best.ys.map((v) => v + dy * q0);
    if (!inside({ xs, ys })) continue;
    const v = shiftScore(xs, ys);
    if (v > shift.s) shift = { dx, dy, s: v };
  }
  if (shift.dx || shift.dy) { best.xs = best.xs.map((v) => v + shift.dx * p0); best.ys = best.ys.map((v) => v + shift.dy * q0); }
  // an outermost row or column without square boundaries, with a board row beyond the opposite edge: one square off
  // (the checkerboard can hardly tell when the board is crowded or its dark squares are finely hatched)
  const e = edgeBands([P.coarse, P.fine, P.tex], best.xs, best.ys);
  let ex = 0, ey = 0;
  if (e.bottom < 0.3 && e.above > 0.7) ey = -1; else if (e.top < 0.3 && e.below > 0.7) ey = 1;
  if (e.last < 0.3 && e.beforeX > 0.7) ex = -1; else if (e.first < 0.3 && e.afterX > 0.7) ex = 1;
  if (ex || ey) {
    const xs = best.xs.map((v) => v + ex * p0), ys = best.ys.map((v) => v + ey * q0);
    if (inside({ xs, ys })) { best.xs = xs; best.ys = ys; }
  }
  if (opts.debug) opts.debug.push({ edgeBands: e, ex, ey });
  const rx = { lines: best.xs }, ry = { lines: best.ys };
  // profile positions are pixel indices (an edge between pixels 9 and 10 peaks at 9.5); the grid is in continuous
  // coordinates (that edge lies at 10) of the snip
  const inv = 1 / sc;
  return {
    xs: rx.lines.map((v) => (v + 0.5) * inv), ys: ry.lines.map((v) => (v + 0.5) * inv),
    ok: best.cov > 0.45 || best.checker.z > 2.5 || best.tchecker.z > 2.5 || best.frame > 0.8,
    score: best.score, checker: best.checker, cov: best.cov, frame: best.frame, tag: best.cand.tag, skew, img,
  };
}

/**
 * Second pass on the grid once the empty squares are known: each inner line is placed where the edge between two
 * empty squares is strongest (pieces next to a boundary no longer pull it), with a blur that flattens hatching
 * when the dark squares are patterned; the lines are then fitted to a straight lattice again.
 * empty: 64 booleans (row-major). Returns { xs, ys } in continuous coordinates of img.
 */
export function refineGrid(img, xs, ys, empty, patterned) {
  const p = (xs[8] - xs[0]) / 8, q = (ys[8] - ys[0]) / 8;
  const b = blur(img, patterned ? Math.max(1.2, p / 14) : 0.7);
  const { w, h, d } = b;
  const fitLine = (pos, period) => {
    const pts = pos.filter((v) => v.ok);
    if (pts.length < 4) return null;
    const n = pts.length, sk = pts.reduce((s, v) => s + v.k, 0), sx = pts.reduce((s, v) => s + v.x, 0);
    const skk = pts.reduce((s, v) => s + v.k * v.k, 0), skx = pts.reduce((s, v) => s + v.k * v.x, 0);
    const den = n * skk - sk * sk;
    const slope = den ? (n * skx - sk * sx) / den : period;
    if (Math.abs(slope - period) > period * 0.05) return null;
    return Array.from({ length: 9 }, (_, k) => (sx - slope * sk) / n + slope * k);
  };
  const along = (vertical) => {
    const lines = vertical ? xs : ys, other = vertical ? ys : xs, period = vertical ? p : q;
    const pos = [], profs = [];
    for (let k = 1; k < 8; k++) {
      const c = lines[k], r = Math.max(2, period * 0.2);
      const lo = Math.floor(c - r), hi = Math.ceil(c + r);
      const prof = new Float32Array(hi - lo + 1);
      let segs = 0;
      for (let s = 0; s < 8; s++) {
        const a = vertical ? empty[s * 8 + k - 1] && empty[s * 8 + k] : empty[(k - 1) * 8 + s] && empty[k * 8 + s];
        if (!a) continue;
        segs++;
        const t0 = Math.round(other[s] + (other[s + 1] - other[s]) * 0.15), t1 = Math.round(other[s + 1] - (other[s + 1] - other[s]) * 0.15);
        for (let t = t0; t < t1; t++) {
          for (let u = lo; u <= hi; u++) {
            // index position u of the edge between pixels u - 1 and u (continuous coordinate u)
            const x = vertical ? u : t, y = vertical ? t : u;
            if (x < 1 || y < 1 || x >= w || y >= h) continue;
            const g = vertical ? Math.abs(d[y * w + x] - d[y * w + x - 1]) : Math.abs(d[y * w + x] - d[(y - 1) * w + x]);
            prof[u - lo] += g;
          }
        }
      }
      if (segs < 2) { pos.push({ k, x: c, ok: false }); continue; }
      let bi = 0;
      for (let i = 1; i < prof.length; i++) if (prof[i] > prof[bi]) bi = i;
      let x = lo + bi;
      if (bi > 0 && bi < prof.length - 1) {
        const a0 = prof[bi - 1], m = prof[bi], b0 = prof[bi + 1], den = a0 - 2 * m + b0;
        if (den < 0) x += Math.max(-0.5, Math.min(0.5, (0.5 * (a0 - b0)) / den));
      }
      pos.push({ k, x, ok: prof[bi] > 0 });
      profs.push({ k, lo, prof });
    }
    const fitted = fitLine(pos, period);
    if (!fitted) return { lines, gain: 1 };
    // how much stronger the boundaries are under the new lines than under the old ones
    const at = (arr) => profs.reduce((sum, { k, lo, prof }) => sum + (prof[Math.round(arr[k]) - lo] || 0), 0);
    return { lines: fitted, gain: at(fitted) / Math.max(1e-6, at(lines)) };
  };
  const vx = along(true), vy = along(false);
  return { xs: vx.lines, ys: vy.lines, gainX: vx.gain, gainY: vy.gain };
}
