// Chess diagrams on a whole page (night pages keep them in their own colours, see reader/page-theme.js): the board
// search of the snip on a copy of the page about 600 pixels wide, up to two boards, only clear ones. The search
// runs in a web worker (app/src/workers/diagram-worker.js), so that turning pages never waits for it.
import { fromRGBA } from './image.js';
import { findBoard } from './locate.js';

/** boards in RGBA pixels → [{ x, y, w, h }] in those pixels, each a little larger than the board (no DOM) */
export function findBoardsInPixels(data, W, H) {
  const img = fromRGBA(data, W, H);
  const out = [];
  for (let n = 0; n < 2; n++) {
    const r = findBoard(img, { skew: 0 });
    // a clear board: a checkerboard, grid lines along most square sides, or a frame around grid lines
    if (!r || !r.ok || !(r.score > 9 || r.checker.z > 2.5 || (r.frame > 0.8 && r.cov > 0.5))) break;
    const side = (r.xs[8] - r.xs[0] + r.ys[8] - r.ys[0]) / 2, pad = side * 0.05;
    const box = { x: r.xs[0] - pad, y: r.ys[0] - pad, w: r.xs[8] - r.xs[0] + 2 * pad, h: r.ys[8] - r.ys[0] + 2 * pad };
    out.push(box);
    // blank it out for the next search
    for (let y = Math.max(0, Math.floor(box.y)); y < Math.min(H, Math.ceil(box.y + box.h)); y++) img.d.fill(1, y * W + Math.max(0, Math.floor(box.x)), y * W + Math.min(W, Math.ceil(box.x + box.w)));
  }
  return out;
}

let worker = null, seq = 0;
const waiting = new Map();
function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('dist/diagram-worker.js', document.baseURI).href, { type: 'module', name: 'diagram' });
    worker.onmessage = (e) => { const w = waiting.get(e.data.id); if (w) { waiting.delete(e.data.id); w(e.data); } };
    worker.onerror = (e) => { console.warn('[chess] diagram worker', e.message); for (const w of waiting.values()) w({ error: e.message || 'worker failed' }); waiting.clear(); worker = null; };
  }
  return worker;
}

/** a message to the diagram worker → its answer (an object with `error` when the worker failed) */
export function askWorker(msg, transfer = []) {
  const id = ++seq;
  return new Promise((resolve) => {
    waiting.set(id, resolve);
    try { getWorker().postMessage({ ...msg, id }, transfer); } catch (e) { waiting.delete(id); resolve({ error: e.message }); }
  });
}

/** boards on a page canvas → [{ x, y, w, h }] in canvas pixels */
export async function findBoardsOnCanvas(canvas) {
  const W = 600, k = W / canvas.width;
  const H = Math.max(1, Math.round(canvas.height * k));
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.drawImage(canvas, 0, 0, W, H);
  const data = ctx.getImageData(0, 0, W, H).data;
  const rects = (await askWorker({ data, width: W, height: H }, [data.buffer])).rects || [];
  return rects.map((b) => ({ x: b.x / k, y: b.y / k, w: b.w / k, h: b.h / k }));
}
