// Grey images and binary masks for the diagram recognition. No DOM: the same code runs in the app and in Node
// (training/bench.js). A grey image is { w, h, d: Float32Array } with 0 = black and 1 = white, a mask is a
// Uint8Array of w * h with 1 = set.

export const gray = (w, h, d) => ({ w, h, d: d || new Float32Array(w * h) });

/** RGBA bytes → grey (luma); transparent pixels count as white paper */
export function fromRGBA(rgba, w, h) {
  const out = gray(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    const a = rgba[j + 3] / 255;
    const l = (0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]) / 255;
    out.d[i] = l * a + (1 - a);
  }
  return out;
}

/** area average when shrinking, bilinear when enlarging */
export function resize(img, nw, nh) {
  nw = Math.max(1, Math.round(nw)); nh = Math.max(1, Math.round(nh));
  if (nw === img.w && nh === img.h) return gray(nw, nh, img.d.slice());
  if (nw < img.w && nh < img.h) return shrink(img, nw, nh);
  const out = gray(nw, nh);
  const sx = img.w / nw, sy = img.h / nh;
  for (let y = 0; y < nh; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    for (let x = 0; x < nw; x++) out.d[y * nw + x] = bilinear(img, (x + 0.5) * sx - 0.5, fy);
  }
  return out;
}

function shrink(img, nw, nh) {
  const { w, h, d } = img;
  const out = gray(nw, nh);
  const sx = w / nw, sy = h / nh;
  // separable box filter with fractional edges
  const tmp = new Float32Array(nw * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < nw; x++) {
      const a = x * sx, b = a + sx;
      let s = 0, i = Math.floor(a);
      for (; i < b; i++) {
        const lo = Math.max(a, i), hi = Math.min(b, i + 1);
        s += d[row + Math.min(w - 1, i)] * (hi - lo);
      }
      tmp[y * nw + x] = s / sx;
    }
  }
  for (let x = 0; x < nw; x++) {
    for (let y = 0; y < nh; y++) {
      const a = y * sy, b = a + sy;
      let s = 0, i = Math.floor(a);
      for (; i < b; i++) {
        const lo = Math.max(a, i), hi = Math.min(b, i + 1);
        s += tmp[Math.min(h - 1, i) * nw + x] * (hi - lo);
      }
      out.d[y * nw + x] = s / sy;
    }
  }
  return out;
}

export function bilinear(img, x, y) {
  const { w, h, d } = img;
  if (x < 0) x = 0; else if (x > w - 1) x = w - 1;
  if (y < 0) y = 0; else if (y > h - 1) y = h - 1;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  const a = d[y0 * w + x0], b = d[y0 * w + x1], c = d[y1 * w + x0], e = d[y1 * w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + e * fx) * fy;
}

/** box blur with radius r (window 2r + 1), edges clamped */
export function boxBlur(img, r) {
  r = Math.max(0, Math.round(r));
  if (!r) return gray(img.w, img.h, img.d.slice());
  const { w, h } = img;
  const tmp = new Float32Array(w * h), out = gray(w, h);
  blurLines(img.d, tmp, w, h, 1, w, r);    // rows
  blurLines(tmp, out.d, h, w, w, 1, r);    // columns
  return out;
}

/** running-sum box filter over `lines` lines of `len` samples (stride `step`, lines `lineStep` apart) */
function blurLines(src, dst, len, lines, step, lineStep, r) {
  const inv = 1 / (2 * r + 1);
  for (let l = 0; l < lines; l++) {
    const o = l * lineStep;
    const first = src[o], last = src[o + (len - 1) * step];
    let s = first * (r + 1);
    for (let k = 1; k <= r; k++) s += k < len ? src[o + k * step] : last;
    for (let i = 0; i < len; i++) {
      dst[o + i * step] = s * inv;
      const add = i + r + 1, sub = i - r;
      s += (add < len ? src[o + add * step] : last) - (sub > 0 ? src[o + sub * step] : first);
    }
  }
}

/** approximate Gaussian: three box passes */
export function blur(img, sigma) {
  if (sigma < 0.4) return gray(img.w, img.h, img.d.slice());
  // box radius for three passes: sigma² = 3 · ((2r + 1)² − 1) / 12
  const r = Math.max(1, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2));
  return boxBlur(boxBlur(boxBlur(img, r), r), r);
}

/** rotate by `deg` degrees around the centre; uncovered pixels become `fill` */
export function rotate(img, deg, fill = 1) {
  const { w, h } = img;
  const out = gray(w, h);
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // inverse mapping: rotate the target point back by −deg
      const dx = x - cx, dy = y - cy;
      const sx = c * dx + s * dy + cx, sy = -s * dx + c * dy + cy;
      out.d[y * w + x] = sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5 ? fill : bilinear(img, sx, sy);
    }
  }
  return out;
}

export function crop(img, x, y, cw, ch) {
  x = Math.max(0, Math.floor(x)); y = Math.max(0, Math.floor(y));
  cw = Math.max(1, Math.min(img.w - x, Math.round(cw))); ch = Math.max(1, Math.min(img.h - y, Math.round(ch)));
  const out = gray(cw, ch);
  for (let yy = 0; yy < ch; yy++) out.d.set(img.d.subarray((y + yy) * img.w + x, (y + yy) * img.w + x + cw), yy * cw);
  return out;
}

/** the quadrilateral cell [x0, x1] × [y0, y1] resampled to n × n (area average when the cell is larger) */
export function cell(img, x0, y0, x1, y1, n) {
  const out = gray(n, n);
  const sx = (x1 - x0) / n, sy = (y1 - y0) / n;
  const ss = Math.max(1, Math.min(4, Math.ceil(Math.max(sx, sy))));   // supersampling per output pixel
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let j = 0; j < ss; j++) {
        const py = y0 + (y + (j + 0.5) / ss) * sy - 0.5;
        for (let i = 0; i < ss; i++) s += bilinear(img, x0 + (x + (i + 0.5) / ss) * sx - 0.5, py);
      }
      out.d[y * n + x] = s / (ss * ss);
    }
  }
  return out;
}

export function percentile(values, q) {
  const n = values.length;
  if (!n) return 0;
  if (n > 4096) return histQuantile(values, q);
  const a = Float32Array.from(values).sort();
  return a[Math.min(n - 1, Math.max(0, Math.floor(q * (n - 1) + 0.5)))];
}

/** quantile of many values in [0, 1] (grey levels, gradients) from a 1024-bin histogram */
export function histQuantile(values, q, lo = 0, hi = 1) {
  const bins = 1024, hist = new Uint32Array(bins), k = bins / (hi - lo);
  for (let i = 0; i < values.length; i++) { const b = Math.floor((values[i] - lo) * k); hist[b < 0 ? 0 : b >= bins ? bins - 1 : b]++; }
  const target = q * (values.length - 1);
  let acc = 0;
  for (let b = 0; b < bins; b++) { acc += hist[b]; if (acc > target) return lo + (b + 0.5) / k; }
  return hi;
}

export function median(values) { return percentile(values, 0.5); }

/** Otsu threshold of values in [lo, hi] */
export function otsu(values, lo = 0, hi = 1, bins = 64) {
  const hist = new Float64Array(bins);
  const n = values.length;
  if (!n) return (lo + hi) / 2;
  for (const v of values) hist[Math.max(0, Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins)))]++;
  let sum = 0;
  for (let i = 0; i < bins; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, best = -1, bestT = bins / 2;
  for (let i = 0; i < bins; i++) {
    wB += hist[i]; if (!wB) continue;
    const wF = n - wB; if (!wF) break;
    sumB += i * hist[i];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; bestT = i + 1; }
  }
  return lo + (bestT / bins) * (hi - lo);
}

// ---------------------------------------------------------------------------------------------- masks
export function dilate(m, w, h, r) {
  if (r <= 0) return m.slice();
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let last = -1e9;
    for (let x = 0; x < w; x++) { if (m[y * w + x]) last = x; tmp[y * w + x] = x - last <= r ? 1 : 0; }
    last = 1e9;
    for (let x = w - 1; x >= 0; x--) { if (m[y * w + x]) last = x; if (last - x <= r) tmp[y * w + x] = 1; }
  }
  for (let x = 0; x < w; x++) {
    let last = -1e9;
    for (let y = 0; y < h; y++) { if (tmp[y * w + x]) last = y; out[y * w + x] = y - last <= r ? 1 : 0; }
    last = 1e9;
    for (let y = h - 1; y >= 0; y--) { if (tmp[y * w + x]) last = y; if (last - y <= r) out[y * w + x] = 1; }
  }
  return out;
}

export function erode(m, w, h, r) {
  if (r <= 0) return m.slice();
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = m[i] ? 0 : 1;
  const d = dilate(inv, w, h, r);
  for (let i = 0; i < w * h; i++) d[i] = d[i] ? 0 : 1;
  return d;
}

/** everything that cannot be reached from the border through unset pixels becomes set */
export function fillHoles(m, w, h) {
  const out = new Uint8Array(w * h).fill(1);
  const stack = [];
  const push = (i) => { if (!m[i] && out[i]) { out[i] = 0; stack.push(i); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const i = stack.pop(), x = i % w, y = (i - x) / w;
    if (x > 0) push(i - 1); if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w); if (y < h - 1) push(i + w);
  }
  return out;
}

/** 4-connected components: { labels: Int32Array, comps: [{ n, x0, y0, x1, y1, sx, sy }] } (label k ↔ comps[k − 1]) */
export function components(m, w, h) {
  const labels = new Int32Array(w * h);
  const comps = [];
  const stack = [];
  for (let s = 0; s < w * h; s++) {
    if (!m[s] || labels[s]) continue;
    const id = comps.length + 1;
    const c = { n: 0, x0: w, y0: h, x1: -1, y1: -1, sx: 0, sy: 0 };
    labels[s] = id; stack.push(s);
    while (stack.length) {
      const i = stack.pop(), x = i % w, y = (i - x) / w;
      c.n++; c.sx += x; c.sy += y;
      if (x < c.x0) c.x0 = x; if (x > c.x1) c.x1 = x; if (y < c.y0) c.y0 = y; if (y > c.y1) c.y1 = y;
      if (x > 0 && m[i - 1] && !labels[i - 1]) { labels[i - 1] = id; stack.push(i - 1); }
      if (x < w - 1 && m[i + 1] && !labels[i + 1]) { labels[i + 1] = id; stack.push(i + 1); }
      if (y > 0 && m[i - w] && !labels[i - w]) { labels[i - w] = id; stack.push(i - w); }
      if (y < h - 1 && m[i + w] && !labels[i + w]) { labels[i + w] = id; stack.push(i + w); }
    }
    comps.push(c);
  }
  return { labels, comps };
}

/** number of holes of a component mask (connected background regions not touching the border) */
export function holes(m, w, h) {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = m[i] ? 0 : 1;
  const { comps } = components(inv, w, h);
  return comps.filter((c) => c.x0 > 0 && c.y0 > 0 && c.x1 < w - 1 && c.y1 < h - 1 && c.n >= 2).length;
}
