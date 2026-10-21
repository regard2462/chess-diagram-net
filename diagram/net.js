// The square classifier of the diagram recognition: a small convolutional network that looks at one cleaned square
// (squares.js: the background, the hatching and the arrows already white) and its difference to the empty square,
// plus a few numbers about the square and the diagram, and gives the probability of each of 13 classes (empty, six
// white and six black pieces). It was trained offline (training/) on thousands of generated diagrams in many
// piece sets, styles and degradations, also piece designs no font on this computer has, so that it recognises a
// book's diagram font it has never seen; the templates of this computer and of the book's corrections still take
// part in the decision (recognize.js). Pure JavaScript, no DOM: it runs in the app and in Node (training/bench.js).
import { N } from './squares.js';

export const CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
/** the numbers next to the picture of a square (squareScalars) */
export const SCALARS = ['light', 'patterned', 'occupied', 'area', 'height', 'width', 'height/tallest', 'centre', 'darkness', 'interior', 'ring', 'strength', 'arrow'];

/** what the squares of one diagram share: the height of its tallest pieces, which colours are patterned */
export function diagramContext(an) {
  const hs = an.squares.filter((s) => s.occ && s.box).map((s) => s.box.h).sort((a, b) => b - a);
  const hmax = !hs.length ? N * 0.9 : hs.length >= 2 ? Math.max(hs[1], 0.94 * hs[0]) : hs[0];
  return { hmax, patterned: an.patterned };
}

/** the numbers of one square (SCALARS) */
export function squareScalars(s, ctx) {
  const b = s.box;
  return [
    s.light ? 1 : 0, (s.light ? ctx.patterned.light : ctx.patterned.dark) ? 1 : 0, s.occ ? 1 : 0, Math.min(1, s.area || 0),
    b ? b.h / N : 0, b ? b.w / N : 0, b ? Math.min(1.3, b.h / ctx.hmax) : 0, b ? ((b.y0 + b.y1) / 2 + (s.shift ? s.shift[1] : 0)) / N : 0,
    s.darkness || 0, s.interior || 0, s.ring || 0, Math.min(8, s.strength || 0) / 8, s.arrow ? 1 : 0,
  ];
}

// ------------------------------------------------------------------------------------------------ weights
const B64 = new Int16Array(128).fill(-1);
'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'.split('').forEach((c, i) => { B64[c.charCodeAt(0)] = i; });
function fromBase64(str) {
  const clean = str.replace(/[^A-Za-z0-9+/]/g, '');
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let o = 0;
  for (let i = 0; i + 3 < clean.length + 3; i += 4) {
    const a = B64[clean.charCodeAt(i)], b = B64[clean.charCodeAt(i + 1)], c = i + 2 < clean.length ? B64[clean.charCodeAt(i + 2)] : -1, d = i + 3 < clean.length ? B64[clean.charCodeAt(i + 3)] : -1;
    if (a < 0 || b < 0) break;
    out[o++] = (a << 2) | (b >> 4);
    if (c >= 0 && o < out.length) out[o++] = ((b & 15) << 4) | (c >> 2);
    if (d >= 0 && o < out.length) out[o++] = ((c & 3) << 6) | d;
  }
  return out.subarray(0, o);
}

/** layers with float weights from the stored ones (int8 per output channel with a scale, float biases) */
function decode(model) {
  return model.layers.map((L) => {
    const q = new Int8Array(fromBase64(L.data).buffer);
    const w = new Float32Array(q.length);
    const per = q.length / L.shape[0];
    for (let o = 0; o < L.shape[0]; o++) for (let k = 0; k < per; k++) w[o * per + k] = q[o * per + k] * L.scale[o];
    return { ...L, w, b: Float32Array.from(L.bias) };
  });
}

let layers = null, modelInfo = null;
/** installs the trained model (net-weights.js); returns false without one */
export function setModel(model) {
  if (!model?.layers?.length) { layers = null; modelInfo = null; return false; }
  layers = decode(model);
  modelInfo = { version: model.version, input: model.input || N };
  return true;
}
export const hasModel = () => !!layers;
export const modelVersion = () => modelInfo?.version || null;

// ------------------------------------------------------------------------------------------------ forward pass
// The feature maps are planes with a border of one zero pixel ((h + 2) × (w + 2) each): the 3 × 3 convolution then
// needs no bounds checks, and its nine weights stay in local variables for a whole plane.

/** 3 × 3 convolution with padding 1 and ReLU: x (cin padded planes of h × w) → out (cout padded planes) */
function conv3(x, cin, h, w, L, out) {
  const cout = L.shape[0], W = L.w, B = L.b;
  const pw = w + 2, plane = pw * (h + 2);
  out.fill(0, 0, cout * plane);
  for (let o = 0; o < cout; o++) {
    const ob = o * plane;
    for (let y = 1; y <= h; y++) out.fill(B[o], ob + y * pw + 1, ob + y * pw + 1 + w);
    for (let c = 0; c < cin; c++) {
      const wb = (o * cin + c) * 9, ib = c * plane;
      const k0 = W[wb], k1 = W[wb + 1], k2 = W[wb + 2], k3 = W[wb + 3], k4 = W[wb + 4], k5 = W[wb + 5], k6 = W[wb + 6], k7 = W[wb + 7], k8 = W[wb + 8];
      for (let y = 1; y <= h; y++) {
        let r0 = ib + (y - 1) * pw, r1 = r0 + pw, r2 = r1 + pw, ro = ob + y * pw + 1;
        for (let xx = 0; xx < w; xx++, r0++, r1++, r2++, ro++) {
          out[ro] += k0 * x[r0] + k1 * x[r0 + 1] + k2 * x[r0 + 2] + k3 * x[r1] + k4 * x[r1 + 1] + k5 * x[r1 + 2] + k6 * x[r2] + k7 * x[r2 + 1] + k8 * x[r2 + 2];
        }
      }
    }
    for (let y = 1; y <= h; y++) for (let j = ob + y * pw + 1, e = j + w; j < e; j++) if (out[j] < 0) out[j] = 0;
  }
}

/** 2 × 2 maximum of padded planes into padded planes of half the size */
function pool2(x, c, h, w, out) {
  const pw = w + 2, plane = pw * (h + 2), h2 = h >> 1, w2 = w >> 1, qw = w2 + 2, qplane = qw * (h2 + 2);
  out.fill(0, 0, c * qplane);
  for (let k = 0; k < c; k++) for (let y = 0; y < h2; y++) {
    let i = k * plane + (2 * y + 1) * pw + 1, o = k * qplane + (y + 1) * qw + 1;
    for (let xx = 0; xx < w2; xx++, i += 2, o++) {
      const a = x[i], b = x[i + 1], cc = x[i + pw], d = x[i + pw + 1];
      const m1 = a > b ? a : b, m2 = cc > d ? cc : d;
      out[o] = m1 > m2 ? m1 : m2;
    }
  }
}

function dense(x, L, relu) {
  const [nout, nin] = L.shape;
  const out = new Float32Array(nout);
  for (let o = 0; o < nout; o++) {
    let s = L.b[o];
    const wo = o * nin;
    for (let i = 0; i < nin; i++) s += L.w[wo + i] * x[i];
    out[o] = relu && s < 0 ? 0 : s;
  }
  return out;
}

/** the input of the network into two padded planes: the cleaned square and the difference, at the size it was
 *  trained with (an area average when that is smaller than N) */
function input(s, size, x) {
  const pw = size + 2, plane = pw * pw;
  x.fill(0, 0, 2 * plane);
  const f = N / size;
  for (let y = 0; y < size; y++) for (let xx = 0; xx < size; xx++) {
    let a = 0, b = 0, n = 0;
    for (let v = Math.floor(y * f); v < Math.ceil((y + 1) * f); v++) for (let u = Math.floor(xx * f); u < Math.ceil((xx + 1) * f); u++) { a += s.clean[v * N + u]; b += s.resid[v * N + u]; n++; }
    x[(y + 1) * pw + xx + 1] = a / n; x[plane + (y + 1) * pw + xx + 1] = b / n;
  }
}

/**
 * classify(squares, ctx) → per square { p: Float32Array(13) (probabilities in CLASSES order), emb: Float32Array }
 * or null without a model. `only`: square indices to look at (the others get null).
 */
export function classify(squares, ctx, only = null) {
  if (!layers) return null;
  const size = modelInfo.input;
  const convs = layers.filter((L) => L.kind === 'conv'), fcs = layers.filter((L) => L.kind === 'fc');
  const room = (size + 2) * (size + 2) * Math.max(2, ...convs.map((L) => L.shape[0]));
  const bufA = new Float32Array(room), bufB = new Float32Array(room);
  return squares.map((s) => {
    if (only && !only.has(s.i)) return null;
    input(s, size, bufA);
    let c = 2, h = size;
    for (const L of convs) { conv3(bufA, c, h, h, L, bufB); c = L.shape[0]; pool2(bufB, c, h, h, bufA); h >>= 1; }
    const sc = s.scalars || squareScalars(s, ctx);
    const z = new Float32Array(c * h * h + sc.length);
    const pw = h + 2, plane = pw * pw;
    for (let k = 0; k < c; k++) for (let y = 0; y < h; y++) for (let xx = 0; xx < h; xx++) z[k * h * h + y * h + xx] = bufA[k * plane + (y + 1) * pw + xx + 1];
    z.set(sc, c * h * h);
    const emb = dense(z, fcs[0], true);
    const logits = dense(emb, fcs[1], false);
    let m = -Infinity; for (const v of logits) if (v > m) m = v;
    let sum = 0; const p = new Float32Array(logits.length);
    for (let k = 0; k < logits.length; k++) { p[k] = Math.exp(logits[k] - m); sum += p[k]; }
    for (let k = 0; k < p.length; k++) p[k] /= sum;
    return { p, emb, logits };
  });
}
