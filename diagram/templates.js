// Templates for the diagram recognition, rendered at run time from local piece sets: the lichess pieces of
// chessground (cburnett, also the app's own 2D board) and the chess glyphs of the fonts installed on this
// computer (Segoe UI Symbol, DejaVu Sans, MS Gothic …). Nothing is downloaded and no font data is shipped: the
// glyphs are drawn with canvas text. Every set is drawn as a few small diagrams with known pieces (white, grey,
// hatched and brown squares, three sizes), which then go through the same square analysis as a snip, so that the
// templates carry the same distortions as the squares they are compared with.
import { fromRGBA } from './image.js';
import { makeTemplateSet, tallest } from './recognize.js';
import { analyzeSquares } from './squares.js';
import { pieceFeatures } from './features.js';
import { loadCachedTemplates, storeCachedTemplates } from './book-memory.js';

/** changes whenever the square analysis or the features change: older cached templates are rebuilt */
export const TEMPLATE_VERSION = 'v3';

const UNI = { K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙', k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
const ROLE = { k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' };
const PIECES = 'KQRBNPkqrbnp';
/** fonts with chess glyphs that may be installed; missing ones fall back to another font and are dropped as duplicates */
export const GLYPH_FONTS = [
  ['segoe', '"Segoe UI Symbol"', 'normal'], ['dejavu', '"DejaVu Sans"', 'normal'], ['dejavu-bold', '"DejaVu Sans"', 'bold'],
  ['gothic', '"MS Gothic"', 'normal'], ['arial-unicode', '"Arial Unicode MS"', 'normal'], ['freeserif', '"FreeSerif"', 'normal'],
  ['noto-symbols', '"Noto Sans Symbols 2"', 'normal'], ['symbola', '"Symbola"', 'normal'], ['quivira', '"Quivira"', 'normal'],
];
const STYLES = {
  grey: { light: '#ffffff', dark: '#b4b4b4' },
  hatch: { light: '#ffffff', dark: 'hatch' },
  brown: { light: '#f0d9b5', dark: '#b58863' },
  'grey-light': { light: '#ffffff', dark: '#d6d6d6' },
};
const SIZES = [30, 44, 60];

/** labels of a template board: every piece on a light and on a dark square, the rest empty */
export function templateLabels() {
  const labels = new Array(64).fill('.');
  [...PIECES].forEach((ch, k) => { labels[2 * k] = ch; labels[2 * k + 1] = ch; });
  return labels.join('');
}

/** mask of everything not reachable from the border through transparent pixels (a filled silhouette) */
function filledAlpha(ctx, size) {
  const a = ctx.getImageData(0, 0, size, size).data;
  const inside = new Uint8Array(size * size).fill(1);
  const stack = [];
  const open = (i) => a[i * 4 + 3] < 110;
  const push = (i) => { if (inside[i] && open(i)) { inside[i] = 0; stack.push(i); } };
  for (let x = 0; x < size; x++) { push(x); push((size - 1) * size + x); }
  for (let y = 0; y < size; y++) { push(y * size); push(y * size + size - 1); }
  while (stack.length) {
    const i = stack.pop(), x = i % size, y = (i - x) / size;
    if (x > 0) push(i - 1); if (x < size - 1) push(i + 1); if (y > 0) push(i - size); if (y < size - 1) push(i + size);
  }
  return inside;
}

/** paints a white silhouette (the filled alpha of `shape`) and then `ink` onto ctx at (x, y): a piece with an
 *  opaque body, as diagram fonts draw them */
function opaquePiece(ctx, x, y, size, shape, ink = shape) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d', { willReadFrequently: true });
  shape(g, size);
  const sil = filledAlpha(g, size);
  const body = g.createImageData(size, size);
  for (let i = 0; i < size * size; i++) if (sil[i]) { body.data[i * 4] = body.data[i * 4 + 1] = body.data[i * 4 + 2] = 255; body.data[i * 4 + 3] = 255; }
  const b = document.createElement('canvas');
  b.width = b.height = size;
  b.getContext('2d').putImageData(body, 0, 0);
  ctx.drawImage(b, x, y);
  if (ink !== shape) { g.clearRect(0, 0, size, size); ink(g, size); }
  ctx.drawImage(c, x, y);
}

let cburnettCache = null;
async function cburnettImages() {
  if (cburnettCache) return cburnettCache;
  const css = await (await fetch(new URL('vendor/chessground/chessground.cburnett.css', document.baseURI))).text();
  const out = {};
  const re = /piece\.(\w+)\.(\w+)\s*\{\s*background-image:\s*url\('(data:image\/svg\+xml;base64,[^']+)'\)/g;
  const loads = [];
  for (const m of css.matchAll(re)) {
    const role = Object.keys(ROLE).find((k) => ROLE[k] === m[1]);
    if (!role) continue;
    const key = m[2] === 'white' ? role.toUpperCase() : role;
    const img = new Image();
    img.src = m[3];
    loads.push(img.decode().then(() => { out[key] = img; }));
  }
  await Promise.all(loads);
  cburnettCache = out;
  return out;
}

/** draws one board of a set: returns { img (grey), xs, ys, labels, set } */
function drawBoard(set, style, p, drawPiece) {
  const pad = Math.round(p * 0.25);
  const W = 8 * p + 2 * pad;
  const c = document.createElement('canvas');
  c.width = c.height = W;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, W);
  const st = STYLES[style];
  for (let r = 0; r < 8; r++) for (let col = 0; col < 8; col++) {
    const x = pad + col * p, y = pad + r * p;
    const light = (r + col) % 2 === 0;
    if (light || st.dark !== 'hatch') { ctx.fillStyle = light ? st.light : st.dark; ctx.fillRect(x, y, p, p); continue; }
    ctx.fillStyle = st.light; ctx.fillRect(x, y, p, p);
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, p, p); ctx.clip();
    ctx.strokeStyle = '#000'; ctx.lineWidth = Math.max(1, p / 36);
    const sp = p / 6;
    for (let k = -8; k <= 8; k++) { ctx.beginPath(); ctx.moveTo(x + k * sp, y + p); ctx.lineTo(x + k * sp + p, y); ctx.stroke(); }
    ctx.restore();
  }
  const labels = templateLabels();
  for (let i = 0; i < 64; i++) {
    const ch = labels[i];
    if (ch !== '.') drawPiece(ctx, pad + (i & 7) * p, pad + (i >> 3) * p, p, ch);
  }
  const data = ctx.getImageData(0, 0, W, W).data;
  return { img: fromRGBA(data, W, W), xs: Array.from({ length: 9 }, (_, k) => pad + k * p), ys: Array.from({ length: 9 }, (_, k) => pad + k * p), labels, set, style };
}

/** a glyph font's pieces: one font size for all, the tallest glyph fills 86 % of the square, common base line */
function glyphDrawer(family, weight, p) {
  const probe = document.createElement('canvas').getContext('2d');
  const font = (px) => `${weight} ${px}px ${family}`;
  probe.font = font(100);
  const boxes = Object.fromEntries(Object.entries(UNI).map(([k, g]) => { const m = probe.measureText(g); return [k, { l: m.actualBoundingBoxLeft, r: m.actualBoundingBoxRight, a: m.actualBoundingBoxAscent, d: m.actualBoundingBoxDescent }]; }));
  const tallest = Math.max(...Object.values(boxes).map((b) => Math.max(b.a + b.d, b.l + b.r)));
  const px = Math.max(4, (100 * p * 0.86) / Math.max(1, tallest));
  const k = px / 100;
  return (ctx, x, y, size, ch) => {
    const black = ch.toLowerCase();
    const b = boxes[black];
    const ox = x + (size - (b.l + b.r) * k) / 2 + b.l * k, oy = y + size * 0.93 - b.d * k;
    const glyph = (g2, c) => { g2.font = font(px); g2.fillStyle = '#000'; g2.textBaseline = 'alphabetic'; g2.fillText(UNI[c], ox - x, oy - y); };
    // the silhouette of the black glyph, the ink of the piece's own glyph (the white glyph is its outline)
    opaquePiece(ctx, x, y, size, (g2) => glyph(g2, black), (g2) => glyph(g2, ch));
  };
}

/** a rough fingerprint of a set's rendering, to drop fonts that fell back to another font */
function fingerprint(board) {
  let h = 0;
  for (let i = 0; i < board.img.d.length; i += 7) h = (h * 31 + Math.round(board.img.d[i] * 15)) >>> 0;
  return h;
}

let templatePromise = null;
/**
 * The template set of this computer (cached for the session and in IndexedDB): { tset, items, key, sets: [names],
 * boards }.
 * The boards are drawn every time (fast); the square analysis of them, which takes a few seconds, runs once per
 * version and set list, a board at a time so that the reader stays responsive. `onProgress(share)` reports it.
 */
export function loadTemplates(onProgress) {
  if (!templatePromise) templatePromise = build(onProgress).catch((e) => { templatePromise = null; throw e; });
  return templatePromise;
}

async function build(onProgress) {
  const boards = [];
  const seen = new Set();
  const cb = await cburnettImages().catch(() => null);
  const jobs = [];
  if (cb && Object.keys(cb).length === 12) jobs.push(['cburnett', (p) => (ctx, x, y, size, ch) => opaquePiece(ctx, x, y, size, (g) => g.drawImage(cb[ch], 0, 0, size, size))]);
  for (const [name, family, weight] of GLYPH_FONTS) jobs.push([name, (p) => glyphDrawer(family, weight, p)]);
  const prints = [];
  for (const [name, drawerFor] of jobs) {
    const first = drawBoard(name, 'grey', 44, drawerFor(44));
    const fp = fingerprint(first);
    if (seen.has(fp)) continue;   // a missing font falls back to one that is already there
    seen.add(fp); prints.push(name + ':' + fp);
    for (const style of Object.keys(STYLES)) for (const p of SIZES) boards.push(style === 'grey' && p === 44 ? first : drawBoard(name, style, p, drawerFor(p)));
  }
  const key = TEMPLATE_VERSION + '|' + prints.join(',');
  const cached = await loadCachedTemplates(key);
  let items = Array.isArray(cached) && cached.length ? cached : null;
  if (!items) {
    items = [];
    for (const [n, b] of boards.entries()) {
      const an = analyzeSquares(b.img, b.xs, b.ys, { arrows: false });
      const hmax = tallest(an.squares.filter((sq) => b.labels[sq.i] !== '.' && sq.box));
      for (const sq of an.squares) {
        const ch = b.labels[sq.i];
        if (ch === '.' || !sq.box) continue;
        const black = ch === ch.toLowerCase();
        items.push({ role: ch.toLowerCase(), black, set: b.set, light: sq.light, feat: pieceFeatures(sq, black, hmax) });
      }
      onProgress?.((n + 1) / boards.length);
      await new Promise((r) => setTimeout(r, 0));
    }
    storeCachedTemplates(key, items);
  }
  return { tset: makeTemplateSet(items), items, key, sets: [...new Set(boards.map((b) => b.set))], boards, cached: !!cached };
}
