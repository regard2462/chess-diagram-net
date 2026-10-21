// Shape features of a piece, independent of its colour: the silhouette (from analyzeSquares) is cropped to its
// box and scaled into 32 × 32 with its aspect kept; from that come an 8 × 8 zoning of the silhouette and of the
// inner details (the ink lines of a white piece, the white lines of a black one), the width and the left and
// right edges per row band (the knight's head looks one way), the top contour per column band (cross, crown
// points, battlements, mitre, ball) and the size of the piece relative to the tallest piece of the same diagram
// (pawns are small in every set, but the sets differ in how much of the square their pieces fill).
import { N } from './squares.js';
import { erode } from './image.js';

const S = 32;

/** silhouette or detail mask (N × N) inside the box → S × S coverage (aspect kept, centred) */
function normalise(mask, box) {
  const out = new Float32Array(S * S);
  const scale = (S - 2) / Math.max(box.w, box.h);
  const ox = (S - box.w * scale) / 2, oy = (S - box.h * scale) / 2;
  const ss = 3;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let v = 0;
      for (let j = 0; j < ss; j++) for (let i = 0; i < ss; i++) {
        const sx = Math.floor(box.x0 + (x + (i + 0.5) / ss - ox) / scale), sy = Math.floor(box.y0 + (y + (j + 0.5) / ss - oy) / scale);
        if (sx >= box.x0 && sx <= box.x1 && sy >= box.y0 && sy <= box.y1) v += mask[sy * N + sx];
      }
      out[y * S + x] = v / (ss * ss);
    }
  }
  return out;
}

export const FEATURE_GROUPS = [
  // name, length, weight
  ['zone', 64, 1.0],
  ['detail', 64, 0.45],
  ['width', 16, 1.0],
  ['left', 16, 0.8],
  ['right', 16, 0.8],
  ['top', 16, 1.0],
  ['geo', 5, 1.4],
];
export const FEATURE_LENGTH = FEATURE_GROUPS.reduce((s, g) => s + g[1], 0);

/** feature vector of an occupied square; `black` chooses which lines inside count as details, `hmax` is the
 *  height of the tallest piece of the diagram (pixels of the normalised square) */
export function pieceFeatures(sq, black, hmax = N * 0.9) {
  const f = new Float32Array(FEATURE_LENGTH);
  const box = sq.box;
  if (!box) return f;
  const sil = normalise(sq.sil, box);
  // details: ink inside the shrunk silhouette (white pieces) or the light lines inside a black piece
  const inner = erode(sq.sil, N, N, Math.max(1, Math.round(Math.min(box.w, box.h) / 14)));
  const det = new Uint8Array(N * N);
  for (let j = 0; j < N * N; j++) if (inner[j]) det[j] = black ? (sq.ink[j] ? 0 : 1) : sq.ink[j];
  const dn = normalise(det, box);
  let o = 0;
  for (let zy = 0; zy < 8; zy++) for (let zx = 0; zx < 8; zx++) {
    let a = 0, b = 0;
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { a += sil[(zy * 4 + y) * S + zx * 4 + x]; b += dn[(zy * 4 + y) * S + zx * 4 + x]; }
    f[o + zy * 8 + zx] = a / 16;
    f[64 + zy * 8 + zx] = b / 16;
  }
  o = 128;
  // rows: width, left and right edge (relative to the centre of the box)
  for (let band = 0; band < 16; band++) {
    let w = 0, l = S, r = -1;
    for (let y = band * 2; y < band * 2 + 2; y++) {
      for (let x = 0; x < S; x++) if (sil[y * S + x] > 0.5) { w++; if (x < l) l = x; if (x > r) r = x; }
    }
    f[o + band] = w / (2 * S);
    f[o + 16 + band] = r >= 0 ? (S / 2 - l) / S : 0;
    f[o + 32 + band] = r >= 0 ? (r + 1 - S / 2) / S : 0;
  }
  o += 48;
  for (let band = 0; band < 16; band++) {
    let top = S;
    for (let x = band * 2; x < band * 2 + 2; x++) for (let y = 0; y < S; y++) if (sil[y * S + x] > 0.5) { if (y < top) top = y; break; }
    f[o + band] = top / S;
  }
  o += 16;
  let area = 0; for (let j = 0; j < N * N; j++) area += sq.sil[j];
  f[o] = Math.min(1.2, box.h / hmax);
  f[o + 1] = Math.min(1.2, box.w / hmax);
  f[o + 2] = Math.min(3, box.h / box.w);
  f[o + 3] = area / (box.w * box.h);
  f[o + 4] = (box.y0 + box.y1) / 2 / N;
  return f;
}
