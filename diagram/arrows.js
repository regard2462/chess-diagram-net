// Arrows and lines drawn over a diagram (a move shown in the book, a direction, a plan): long, thin, straight strokes
// that run across squares, mostly with a head at one end. They are searched on the evidence of the squares put
// together into one board (the hatching of the dark squares is gone there, and every square's window is put back where
// it lies in the snip): a Hough transform over the thin strokes, then each line refitted to its pixels and walked
// along its length. A line counts when it is long, straight, nearly unbroken, crosses at least one square boundary
// and runs free of other strokes on both sides (the outline of a piece, such as the base of a rook, always has a
// second line next to it). The head is the widening at one end. The result are masks per square: those pixels are no
// evidence of a piece, so that a square crossed by a line is not taken for a piece and a piece with a line through it
// keeps its own shape. Inside the dark body of a black piece a line is invisible and nothing is taken away.
import { erode, dilate, components } from './image.js';

const T = 180;
const COS = new Float32Array(T), SIN = new Float32Array(T);
for (let t = 0; t < T; t++) { COS[t] = Math.cos((t * Math.PI) / T); SIN[t] = Math.sin((t * Math.PI) / T); }

/**
 * findArrows(evid, squares, N, edge) → { segments, masks: Map(square index → Uint8Array N·N), board (debug) }
 * evid: per square the evidence window (N × N) in units of its threshold (1 = threshold); squares: [{ i, r, c, shift }]
 * edge: pixels at the outer edge of the board that never count (frames).
 * Segments are in board pixels (N per square, origin at the board's top left corner).
 */
export function findArrows(evid, squares, N, edge = 3, debug = null) {
  const P = 8, BW = 8 * N + 2 * P;
  const R = new Float32Array(BW * BW);
  for (const s of squares) {
    const a = evid[s.i];
    const ox = s.c * N + Math.round(s.shift[0]) + P, oy = s.r * N + Math.round(s.shift[1]) + P;
    // nothing from outside the board or from the band along its edge (a frame), wherever the shift put the window
    const ya = s.r === 0 ? Math.max(1, edge - Math.floor(s.shift[1])) : 1, yb = s.r === 7 ? Math.min(N - 1, N - edge - Math.ceil(s.shift[1])) : N - 1;
    const xa = s.c === 0 ? Math.max(1, edge - Math.floor(s.shift[0])) : 1, xb = s.c === 7 ? Math.min(N - 1, N - edge - Math.ceil(s.shift[0])) : N - 1;
    for (let y = ya; y < yb; y++) {
      const Y = oy + y;
      if (Y < 0 || Y >= BW) continue;
      for (let x = xa; x < xb; x++) {
        const X = ox + x;
        if (X < 0 || X >= BW) continue;
        const v = a[y * N + x], j = Y * BW + X;
        if (v > R[j]) R[j] = v;
      }
    }
  }
  const B = new Uint8Array(BW * BW);
  let nB = 0;
  for (let j = 0; j < BW * BW; j++) if (R[j] > 1) { B[j] = 1; nB++; }
  const none = { segments: [], masks: new Map(), board: { R, B, BW, P, mask: new Uint8Array(BW * BW) } };
  if (nB < N) return none;
  // thick parts survive an opening (black bodies, filled heads); thin strokes are what is left of the evidence;
  // bodies are the large thick parts (black pieces), a line is invisible inside them
  const rad = Math.max(2, Math.round(N / 16));
  const thick = dilate(erode(B, BW, BW, rad), BW, BW, rad);
  const thickNear = dilate(thick, BW, BW, 1);
  const thin = new Uint8Array(BW * BW);
  const pts = [];
  for (let j = 0; j < BW * BW; j++) if (B[j] && !thickNear[j]) { thin[j] = 1; pts.push(j); }
  const body = new Uint8Array(BW * BW);
  {
    const { labels, comps } = components(thick, BW, BW);
    const big = comps.map((c) => c.n >= 0.05 * N * N);
    for (let j = 0; j < BW * BW; j++) if (labels[j] && big[labels[j] - 1]) body[j] = 1;
  }
  const bodyNear = dilate(body, BW, BW, 1);
  if (pts.length < N * 0.6) return none;
  // Hough transform over the thin pixels
  const RMAX = Math.ceil(BW * 1.4143), NR = 2 * RMAX + 1;
  const acc = new Float32Array(T * NR);
  for (const j of pts) {
    const x = j % BW, y = (j - x) / BW;
    for (let t = 0; t < T; t++) acc[t * NR + Math.round(x * COS[t] + y * SIN[t]) + RMAX]++;
  }
  // peaks: the sum over three neighbouring distances, apart by more than 3° or 4 pixels (a line near the vertical
  // shows up at both ends of the angle range, with the distance negated)
  const minVotes = N * 0.9;
  const cand = [];
  for (let t = 0; t < T; t++) {
    for (let r = 1; r < NR - 1; r++) {
      const v = acc[t * NR + r - 1] + acc[t * NR + r] + acc[t * NR + r + 1];
      if (v >= minVotes) cand.push({ t, rho: r - RMAX, v });
    }
  }
  cand.sort((a, b) => b.v - a.v);
  const peaks = [];
  const near = (p, c) => {
    const dt = Math.abs(p.t - c.t);
    if (dt <= 3) return Math.abs(p.rho - c.rho) <= 4;
    if (T - dt <= 3) return Math.abs(p.rho + c.rho) <= 4;
    return false;
  };
  for (const c of cand) {
    if (peaks.some((p) => near(p, c))) continue;
    peaks.push(c);
    if (peaks.length >= 40) break;
  }
  const get = (m, x, y) => { const X = Math.round(x), Y = Math.round(y); return X >= 0 && Y >= 0 && X < BW && Y < BW ? m[Y * BW + X] : 0; };
  const used = new Uint8Array(BW * BW);
  const segments = [];
  for (const pk of peaks) {
    const nx = COS[pk.t], ny = SIN[pk.t];
    const seg0 = fitLine(pk.rho * nx, pk.rho * ny, -ny, nx, thin, used, BW, N, P);
    if (!seg0) continue;
    const seg = walk(seg0, B, thin, body, used, BW, N, P, get);
    if (debug) debug.push({ t: pk.t, rho: pk.rho, v: pk.v, seg: seg && { x0: +seg.x0.toFixed(1), y0: +seg.y0.toFixed(1), x1: +seg.x1.toFixed(1), y1: +seg.y1.toFixed(1), len: +seg.len.toFixed(1), cover: +seg.cover.toFixed(2), rms: +seg.rms.toFixed(2), side: +seg.side.toFixed(2), w: seg.w, fresh: +seg.fresh.toFixed(2), heads: seg.heads.length, centres: centresOn(seg, N, P) } });
    if (!seg) continue;
    if (seg.len < 0.6 * N || seg.cover < 0.72 || seg.rms > 1.3 + seg.w / 4 || seg.side > 0.45 || seg.w > 0.14 * N || seg.fresh < 0.5) continue;
    if (!crossesBoundary(seg, N, P) || centresOn(seg, N, P) < 2) continue;
    if (segments.some((o) => sameLine(o, seg, N))) continue;
    // grid lines: an axis-parallel line on a square boundary is the board's own
    const ang = Math.abs(Math.atan2(seg.uy, seg.ux)) % (Math.PI / 2);
    if (Math.min(ang, Math.PI / 2 - ang) < 0.035) {
      const vertical = Math.abs(seg.ux) < Math.abs(seg.uy);
      const pos = vertical ? (seg.x0 + seg.x1) / 2 - P : (seg.y0 + seg.y1) / 2 - P;
      if (Math.abs(pos - Math.round(pos / N) * N) < 3) continue;
    }
    segments.push(seg);
    // its pixels are taken: another peak of the same line finds nothing new there
    band(seg, seg.w / 2 + 2, BW, (j) => { used[j] = 1; });
    for (const hd of seg.heads) headPixels(hd, BW, (j) => { used[j] = 1; });
  }
  // masks: the line itself (thin pixels only, never a black body) and the head
  const mask = new Uint8Array(BW * BW);
  for (const s of segments) {
    band(s, s.w / 2 + 1.3, BW, (j) => { if (B[j] && !bodyNear[j]) mask[j] = 1; });
    for (const hd of s.heads) headPixels(hd, BW, (j) => { if (B[j] && !body[j]) mask[j] = 1; });
  }
  const masks = new Map();
  for (const s of squares) {
    const ox = s.c * N + Math.round(s.shift[0]) + P, oy = s.r * N + Math.round(s.shift[1]) + P;
    let m = null;
    for (let y = 0; y < N; y++) {
      const Y = oy + y;
      if (Y < 0 || Y >= BW) continue;
      for (let x = 0; x < N; x++) {
        const X = ox + x;
        if (X < 0 || X >= BW || !mask[Y * BW + X]) continue;
        (m ||= new Uint8Array(N * N))[y * N + x] = 1;
      }
    }
    if (m) masks.set(s.i, m);
  }
  const out = segments.map((s) => ({ x0: s.x0 - P, y0: s.y0 - P, x1: s.x1 - P, y1: s.y1 - P, ux: s.ux, uy: s.uy, len: s.len, w: s.w, heads: s.heads.map((h) => ({ x: h.x - P, y: h.y - P, ux: h.ux, uy: h.uy, len: h.len, width: h.width })) }));
  return { segments: out, masks, board: { R, B, BW, P, mask } };
}

/** the pixels of a band along a segment (half width `half`) */
function band(s, half, BW, fn) {
  const minX = Math.max(0, Math.floor(Math.min(s.x0, s.x1) - half - 2)), maxX = Math.min(BW - 1, Math.ceil(Math.max(s.x0, s.x1) + half + 2));
  const minY = Math.max(0, Math.floor(Math.min(s.y0, s.y1) - half - 2)), maxY = Math.min(BW - 1, Math.ceil(Math.max(s.y0, s.y1) + half + 2));
  for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) {
    const ax = x - s.x0, ay = y - s.y0;
    const along = ax * s.ux + ay * s.uy, perp = Math.abs(-ax * s.uy + ay * s.ux);
    if (along >= -1.5 && along <= s.len + 1.5 && perp <= half) fn(y * BW + x);
  }
}

/** the pixels of a head: a triangle from the tip back along the line, as wide as the head at its back */
function headPixels(hd, BW, fn) {
  const r = hd.len + hd.width / 2 + 3;
  for (let y = Math.max(0, Math.floor(hd.y - r)); y <= Math.min(BW - 1, Math.ceil(hd.y + r)); y++) {
    for (let x = Math.max(0, Math.floor(hd.x - r)); x <= Math.min(BW - 1, Math.ceil(hd.x + r)); x++) {
      const hx = x - hd.x, hy = y - hd.y;
      const back = -(hx * hd.ux + hy * hd.uy), side = Math.abs(-hx * hd.uy + hy * hd.ux);
      if (back < -1.5 || back > hd.len + 1) continue;
      if (side <= 1.5 + (hd.width / 2) * Math.min(1, Math.max(0, back) / Math.max(1, hd.len))) fn(y * BW + x);
    }
  }
}

/** a line through (px, py) along (dx, dy) refitted to the thin pixels near it (twice, the second time closer), with
 *  the share of them not yet taken by another line */
function fitLine(px, py, dx, dy, thin, used, BW, N, P) {
  let cx = px, cy = py, ux = dx, uy = dy;
  let pts = null;
  for (const tol of [2.5, 1.6]) {
    const xs = [], ys = [];
    const L = BW * 1.5;
    for (let u = -L; u <= L; u++) {
      const lx = cx + u * ux, ly = cy + u * uy;
      if (lx < P - 2 || ly < P - 2 || lx > BW - P + 2 || ly > BW - P + 2) continue;
      for (let o = -Math.ceil(tol); o <= Math.ceil(tol); o++) {
        const X = Math.round(lx - o * uy), Y = Math.round(ly + o * ux);
        if (X < 0 || Y < 0 || X >= BW || Y >= BW || !thin[Y * BW + X]) continue;
        const perp = Math.abs(-(X - cx) * uy + (Y - cy) * ux);
        if (perp > tol) continue;
        xs.push(X); ys.push(Y);
      }
    }
    if (xs.length < N * 0.5) return null;
    // the pixels of one line, counted once
    const seen = new Set();
    const fx = [], fy = [];
    for (let k = 0; k < xs.length; k++) { const j = ys[k] * BW + xs[k]; if (seen.has(j)) continue; seen.add(j); fx.push(xs[k]); fy.push(ys[k]); }
    let mx = 0, my = 0;
    for (let k = 0; k < fx.length; k++) { mx += fx[k]; my += fy[k]; }
    mx /= fx.length; my /= fx.length;
    let sxx = 0, syy = 0, sxy = 0;
    for (let k = 0; k < fx.length; k++) { const a = fx[k] - mx, b = fy[k] - my; sxx += a * a; syy += b * b; sxy += a * b; }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    cx = mx; cy = my; ux = Math.cos(th); uy = Math.sin(th);
    pts = { fx, fy };
  }
  let fresh = 0, rms = 0;
  for (let k = 0; k < pts.fx.length; k++) {
    if (!used[pts.fy[k] * BW + pts.fx[k]]) fresh++;
    const perp = -(pts.fx[k] - cx) * uy + (pts.fy[k] - cy) * ux;
    rms += perp * perp;
  }
  return { cx, cy, ux, uy, fresh: fresh / pts.fx.length, rms: Math.sqrt(rms / pts.fx.length), n: pts.fx.length };
}

/** walks the fitted line: the longest stretch of stroke along it (gaps only at square boundaries and inside black
 *  bodies), its width, how free it runs and the heads at its ends */
function walk(f, B, thin, thick, used, BW, N, P, get) {
  const { cx, cy, ux, uy } = f;
  const L = Math.ceil(BW * 1.5);
  const nearBoundary = (x, y) => {
    const bx = (x - P) / N, by = (y - P) / N;
    return Math.abs(bx - Math.round(bx)) * N < 4 || Math.abs(by - Math.round(by)) * N < 4;
  };
  const state = [], freshAt = [];
  for (let u = -L; u <= L; u++) {
    const x = cx + u * ux, y = cy + u * uy;
    if (x < P - 1 || y < P - 1 || x > BW - P || y > BW - P) { state.push(-1); freshAt.push(0); continue; }
    let s = 0, fresh = 0;
    for (let o = -1.5; o <= 1.5; o += 0.5) {
      const X = x - o * uy, Y = y + o * ux;
      if (get(thin, X, Y)) { s = 2; if (!get(used, X, Y)) fresh = 1; continue; }
      if (get(B, X, Y) && s < 2) { s = get(thick, X, Y) ? 1 : 2; if (s === 2 && !get(used, X, Y)) fresh = 1; }
    }
    freshAt.push(fresh);
    if (!s && nearBoundary(x, y)) s = 3;   // a seam between two squares
    state.push(s);
  }
  // stretches: stroke (2), bodies (1) and seams (3) in between are bridged, a few empty steps too
  let best = null, cur = null, gap = 0;
  const close = () => { if (cur && (!best || cur.stroke > best.stroke)) best = cur; cur = null; gap = 0; };
  for (let k = 0; k < state.length; k++) {
    const s = state[k];
    if (s === 2) { if (!cur) cur = { a: k, b: k, stroke: 0 }; cur.b = k; cur.stroke++; gap = 0; }
    else if (s === 0 || s === -1) { if (cur && ++gap > 0.06 * N) close(); }
  }
  close();
  if (!best) return null;
  let body = 0, seam = 0, fresh = 0;
  for (let k = best.a; k <= best.b; k++) { if (state[k] === 1) body++; else if (state[k] === 3) seam++; else if (state[k] === 2 && freshAt[k]) fresh++; }
  const u0 = best.a - L, u1 = best.b - L;
  const x0 = cx + u0 * ux, y0 = cy + u0 * uy, len = u1 - u0;
  const cover = best.stroke / Math.max(1, len + 1 - body - seam);
  // width across the stroke, and strokes running beside it
  const widths = [];
  let sideHits = 0, sideN = 0;
  for (let s = 0.1 * len; s <= 0.9 * len; s += 2) {
    const px = x0 + s * ux, py = y0 + s * uy;
    if (get(thick, px, py)) continue;
    let w = 0;
    if (get(B, px, py)) {
      w = 1;
      for (let o = 1; o < 8 && get(B, px - o * uy, py + o * ux); o++) w++;
      for (let o = 1; o < 8 && get(B, px + o * uy, py - o * ux); o++) w++;
      widths.push(w);
    }
    const wh = Math.max(2, w / 2 + 2);
    let left = 0, right = 0;
    for (let o = wh; o <= wh + 3; o++) {
      if (get(B, px - o * uy, py + o * ux)) left = 1;
      if (get(B, px + o * uy, py - o * ux)) right = 1;
    }
    sideN++;
    if (left || right) sideHits++;
  }
  if (widths.length < 3) return null;
  widths.sort((a, b) => a - b);
  const w = widths[widths.length >> 1];
  const seg = { x0, y0, x1: x0 + len * ux, y1: y0 + len * uy, ux, uy, len, w, cover, rms: f.rms, fresh: fresh / Math.max(1, best.stroke), side: sideN ? sideHits / sideN : 1, heads: [] };
  // heads: the stroke widens towards an end (a filled triangle or two short strokes); the first steps behind the tip
  // do not count (a piece the arrow points at may touch it there)
  for (const end of [1, -1]) {
    const ex = end > 0 ? seg.x1 : seg.x0, ey = end > 0 ? seg.y1 : seg.y0;
    const fx = end * ux, fy = end * uy;   // pointing out of the line at this end
    let tip = 0;
    for (let s = 1; s <= 0.3 * N; s++) { if (get(B, ex + s * fx, ey + s * fy)) tip = s; else if (s - tip > 1) break; }
    const tx = ex + tip * fx, ty = ey + tip * fy;
    let spread = 0, at = 0;
    for (let s = Math.round(0.08 * N); s <= 0.42 * N; s++) {
      const px = tx - s * fx, py = ty - s * fy;
      let m = 0;
      for (let o = 1; o <= 0.2 * N; o++) {
        const a = get(B, px - o * fy, py + o * fx) && !get(thick, px - o * fy, py + o * fx), b = get(B, px + o * fy, py - o * fx) && !get(thick, px + o * fy, py - o * fx);
        if (a || b) m = o; else if (o - m > 1) break;
      }
      if (m > spread) { spread = m; at = s; }
    }
    if (spread >= Math.max(w / 2 + 2, 0.9 * w + 1)) seg.heads.push({ x: tx, y: ty, ux: fx, uy: fy, len: at + 2, width: 2 * spread });
  }
  if (seg.heads.length === 2) { seg.heads.sort((a, b) => b.width - a.width); seg.heads.length = 1; }
  return seg;
}

/** two segments on one line (another peak of the same arrow) */
function sameLine(a, b, N) {
  const cross = Math.abs(a.ux * b.uy - a.uy * b.ux);
  if (cross > 0.14) return false;
  const mx = (b.x0 + b.x1) / 2 - a.x0, my = (b.y0 + b.y1) / 2 - a.y0;
  if (Math.abs(-mx * a.uy + my * a.ux) > 5) return false;
  const along = mx * a.ux + my * a.uy;
  return along > -N * 0.3 && along < a.len + N * 0.3;
}

/** how many square centres the line passes close by (a book draws its arrows from the middle of a square to the
 *  middle of another; the outlines of pieces never run through the middle of two squares) */
function centresOn(s, N, P) {
  let n = 0;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const ax = P + (c + 0.5) * N - s.x0, ay = P + (r + 0.5) * N - s.y0;
    const along = ax * s.ux + ay * s.uy, perp = Math.abs(-ax * s.uy + ay * s.ux);
    if (perp < 0.18 * N && along > -0.65 * N && along < s.len + 0.65 * N) n++;
  }
  return n;
}

/** the line runs from one square into another */
function crossesBoundary(s, N, P) {
  const sq = (x, y) => [Math.floor((x - P) / N), Math.floor((y - P) / N)];
  const a = sq(s.x0 + s.ux * 2, s.y0 + s.uy * 2), b = sq(s.x1 - s.ux * 2, s.y1 - s.uy * 2);
  return a[0] !== b[0] || a[1] !== b[1];
}
