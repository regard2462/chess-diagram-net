// Diagrams of real books for the square classifier: the pages of a book rendered by training/pages.py go
// through the app's own page search (find-boards.js), every board found is cut out with some margin like a snip,
// analysed (board, grid, squares) and read by the current recogniser. Every square with something on it (and a
// share of the clearly empty ones) becomes a sample in the format of extract.js, at the network's input size; its
// class is the current network's reading until label.py replaces it with a checked one.
//   node training/scan-books.js <page folder> <out prefix> --book <id> [--templates <dump>] [--seed n]
//        [--empty 0.12] [--max-boards 9]
// Writes <out>.x (uint8, 2 × S × S), <out>.s (float32 scalars), <out>.y (uint8 class), <out>.l (float32 13 logits),
// <out>.json: { n, N: S, scalars, classes, meta: [[diagram, square, light, occ]], diagrams: [{ page, w, h, zoom,
// rect, crop, xs, ys, kept (64 chars: 1 = sample, 0 = left out as empty), read (64 chars, the network alone),
// placement, grid, confidence, ms }] }
const fs = require('fs');
const os = require('os');
const path = require('path');
const { pathToFileURL } = require('url');
const png = require('./lib/png-lite');

const args = process.argv.slice(2);
const [folder, outPrefix] = args;
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
if (!folder || !outPrefix) { console.log('usage: node training/scan-books.js <page folder> <out prefix> --book <id> [--templates <dump>]'); process.exit(1); }
// with --low: below normal priority, other work on this computer goes first
if (process.argv.includes('--low')) try { os.setPriority(0, 10); } catch {}

(async () => {
  const D = path.join(__dirname, '..', 'diagram');
  const rec = await import(pathToFileURL(path.join(D, 'recognize.js')).href);
  const im = await import(pathToFileURL(path.join(D, 'image.js')).href);
  const net = await import(pathToFileURL(path.join(D, 'net.js')).href);
  const { findBoard } = await import(pathToFileURL(path.join(D, 'locate.js')).href);
  const { N } = await import(pathToFileURL(path.join(D, 'squares.js')).href);
  const S = +opt('--size', 32);
  const book = opt('--book', path.basename(folder));
  const emptyShare = +opt('--empty', 0.12);
  const maxBoards = +opt('--max-boards', 9);
  let seed = +opt('--seed', 1);
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  let templates = null;
  if (opt('--templates', '')) {
    const dump = JSON.parse(fs.readFileSync(opt('--templates', ''), 'utf8'));
    templates = rec.buildTemplates(dump.boards.map((b) => ({ ...b, img: { w: b.w, h: b.h, d: Float32Array.from(Buffer.from(b.d, 'base64'), (v) => v / 255) } })));
  }
  const pages = JSON.parse(fs.readFileSync(path.join(folder, 'pages.json'), 'utf8'));
  // the network's input: the area average of the N × N square down to S × S (net.js input())
  const pool = (src, off, scale, bias, out, o) => {
    const f = N / S;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      let a = 0, n = 0;
      for (let v = Math.floor(y * f); v < Math.ceil((y + 1) * f); v++) for (let u = Math.floor(x * f); u < Math.ceil((x + 1) * f); u++) { a += Math.max(-1, Math.min(1, src[v * N + u])); n++; }
      out[o + y * S + x] = Math.round(Math.max(0, Math.min(255, (a / n + bias) * scale)));
    }
  };
  /** places on a page that may hold a board: the ink of the page (darker than its paper) joined over gaps of two
   *  pixels, regions about as wide as high and at least an eighth of the page wide (a board with its frame,
   *  coordinates or hatched squares forms one region; a paragraph of text is much wider than high) */
  const candidates = (small) => {
    const { w, h, d } = small;
    const paper = im.histQuantile(d, 0.9);
    const t = paper - Math.max(0.2, 0.35 * (paper - im.histQuantile(d, 0.01)));
    const m = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) m[i] = d[i] < t ? 1 : 0;
    const { comps } = im.components(im.dilate(m, w, h, 2), w, h);
    return comps.map((c) => ({ x: c.x0, y: c.y0, w: c.x1 - c.x0 + 1, h: c.y1 - c.y0 + 1, n: c.n }))
      .filter((c) => c.w >= w * 0.12 && c.h >= w * 0.12 && c.w / c.h > 0.7 && c.w / c.h < 1.45)
      .sort((a, b) => b.w * b.h - a.w * a.h);
  };
  const X = [], SC = [], Y = [], LG = [], meta = [], diagrams = [];
  const t0 = Date.now();
  let found = 0, analysed = 0;
  const tm = { search: 0, analyse: 0, net: 0 };
  for (const pg of pages) {
    const t1 = Date.now();
    const raw = png.decode(fs.readFileSync(path.join(folder, pg.file)));
    const page = im.fromRGBA(raw.data, raw.width, raw.height);
    // the page search of the app: a copy 600 pixels wide, clear boards only, each blanked out for the next search
    const k = 600 / page.w;
    const small = im.resize(page, 600, Math.max(1, Math.round(page.h * k)));
    const rects = [];
    for (const c of candidates(small).slice(0, maxBoards)) {
      // the candidate with a margin, searched like a snip
      const pad = Math.max(c.w, c.h) * 0.1;
      const cx = Math.max(0, Math.floor(c.x - pad)), cy = Math.max(0, Math.floor(c.y - pad));
      const part = im.crop(small, cx, cy, Math.min(small.w - cx, c.w + 2 * pad), Math.min(small.h - cy, c.h + 2 * pad));
      // lenient here (at 600 pixels a small board's hatching is a grey blur), strict on the full page below
      const r = findBoard(part, { skew: 0 });
      if (!r || !r.ok || !(r.score > 7 || r.checker.z > 2 || r.cov > 0.6 || r.frame > 0.6)) continue;
      const box = { x: cx + r.xs[0], y: cy + r.ys[0], w: r.xs[8] - r.xs[0], h: r.ys[8] - r.ys[0] };
      if (rects.some((o) => Math.abs(o.x - box.x) < box.w / 4 && Math.abs(o.y - box.y) < box.h / 4)) continue;
      rects.push(box);
    }
    tm.search += Date.now() - t1;
    for (const r of rects) {
      found++;
      const side = (r.w + r.h) / 2 / k;
      // a snip as a reader would draw it: the board with a margin of 4 to 20 % of its side on each edge
      const m = () => side * (0.04 + 0.16 * rnd());
      const x0 = Math.max(0, Math.floor(r.x / k - m())), y0 = Math.max(0, Math.floor(r.y / k - m()));
      const x1 = Math.min(page.w, Math.ceil((r.x + r.w) / k + m())), y1 = Math.min(page.h, Math.ceil((r.y + r.h) / k + m()));
      const crop = im.crop(page, x0, y0, x1 - x0, y1 - y0);
      const ts = Date.now();
      let d = null, res = null;
      try {
        d = rec.analyzeDiagram(crop);
        // the app's whole decision only on request (it doubles the time)
        if (d?.an && args.includes('--read')) res = rec.recognize(crop, { templates, analysis: d });
      } catch (e) { console.log('ERROR', pg.file, e.message); }
      tm.analyse += Date.now() - ts;
      if (!d?.an || (args.includes('--read') && !res?.ok)) continue;
      // a clear board at the page's resolution: a checkerboard, grid lines along most square sides, or a frame
      const f = d.found;
      if (!(f.score > 9 || f.checker.z > 2.5 || f.cov > 0.7 || (f.frame > 0.8 && f.cov > 0.5))) continue;
      analysed++;
      const ctx = net.diagramContext(d.an);
      const tn = Date.now();
      const all = net.classify(d.an.squares, ctx);
      tm.net += Date.now() - tn;
      const di = diagrams.length;
      let kept = '', read = '';
      for (const s of d.an.squares) {
        const o = all ? all[s.i] : null;
        const logits = o ? Array.from(o.logits) : new Array(13).fill(0);
        let best = 0; for (let c = 1; c < 13; c++) if (logits[c] > logits[best]) best = c;
        read += net.CLASSES[best];
        const keep = best !== 0 || s.occ || s.arrow || (o && o.p[0] < 0.9) || rnd() < emptyShare;
        kept += keep ? '1' : '0';
        if (!keep) continue;
        const x = new Uint8Array(2 * S * S);
        pool(s.clean, 0, 255, 0, x, 0);
        pool(s.resid, 0, 127.5, 1, x, S * S);
        X.push(x); SC.push(net.squareScalars(s, ctx)); Y.push(best); LG.push(logits);
        meta.push([di, s.i, s.light ? 1 : 0, s.occ ? 1 : 0]);
      }
      diagrams.push({ page: pg.page, w: pg.w, h: pg.h, zoom: +pg.zoom.toFixed(5), rect: [r.x / k, r.y / k, r.w / k, r.h / k].map((v) => +v.toFixed(1)), crop: [x0, y0, x1 - x0, y1 - y0],
        xs: d.board.xs.map((v) => +v.toFixed(2)), ys: d.board.ys.map((v) => +v.toFixed(2)), kept, read,
        ...(res ? { placement: res.placement, grid: res.grid, confidence: +res.confidence.toFixed(4), uncertain: res.uncertain.length } : {}), ms: Date.now() - ts });
    }
  }
  fs.writeFileSync(outPrefix + '.x', Buffer.concat(X.map((x) => Buffer.from(x.buffer))));
  fs.writeFileSync(outPrefix + '.s', Buffer.concat(SC.map((s) => Buffer.from(Float32Array.from(s).buffer))));
  fs.writeFileSync(outPrefix + '.y', Buffer.from(Uint8Array.from(Y)));
  fs.writeFileSync(outPrefix + '.l', Buffer.concat(LG.map((l) => Buffer.from(Float32Array.from(l).buffer))));
  fs.writeFileSync(outPrefix + '.json', JSON.stringify({ n: Y.length, N: S, scalars: net.SCALARS.length, classes: net.CLASSES, book, meta, diagrams }));
  console.log(`${book}: ${pages.length} pages, ${found} boards found, ${analysed} analysed, ${Y.length} squares kept, ${((Date.now() - t0) / 1000).toFixed(0)} s (search ${(tm.search / 1000).toFixed(0)}, analysis ${(tm.analyse / 1000).toFixed(0)}, network ${(tm.net / 1000).toFixed(0)})`);
})();
