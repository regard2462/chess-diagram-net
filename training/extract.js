// Training data for the square classifier of the diagram recognition (diagram/net.js): the diagrams
// of `python training/gen_chess.py book …` go through the app's own analysis (board, grid, squares, arrows) and
// every square becomes a sample: the cleaned square and its difference to the empty square (2 × 48 × 48 bytes, or
// the area average at --size, the network's input size), a few numbers about it (net.js SCALARS) and its piece.
// Squares that are clearly empty are kept only in part.
//   node training/extract.js <folder> <out prefix> [--shard i/n] [--empty 0.2] [--truth truth-1.json,…] [--size 32]
// Writes <out>.x (uint8), <out>.s (float32), <out>.y (uint8 class, net.js CLASSES), <out>.json (per sample: diagram,
// set, style, degradation). Truth entries without a board rectangle (datasets) are checked on the occupied squares.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { readImage } = require('./lib/image-file');

const args = process.argv.slice(2);
const [folder, outPrefix] = args;
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
if (!folder || !outPrefix) { console.log('usage: node training/extract.js <folder> <out prefix> [--shard i/n] [--empty 0.2]'); process.exit(1); }

(async () => {
  const D = path.join(__dirname, '..', 'diagram');
  const rec = await import(pathToFileURL(path.join(D, 'recognize.js')).href);
  const im = await import(pathToFileURL(path.join(D, 'image.js')).href);
  const net = await import(pathToFileURL(path.join(D, 'net.js')).href);
  const { N } = await import(pathToFileURL(path.join(D, 'squares.js')).href);
  let truth = [];
  const only = opt('--truth', '') ? opt('--truth', '').split(',') : null;
  for (const f of fs.readdirSync(folder).filter((f) => /^truth(-\d+)?\.json$/.test(f) && (!only || only.includes(f))).sort()) truth.push(...JSON.parse(fs.readFileSync(path.join(folder, f), 'utf8')));
  const [si, sn] = (opt('--shard', '0/1')).split('/').map(Number);
  truth = truth.filter((t, i) => i % sn === si);
  const emptyShare = +opt('--empty', 0.2);
  const SZ = +opt('--size', N);
  // the area average of the N × N square down to S × S (net.js input()), as bytes
  const pool = (src, scale, bias, out, o) => {
    const f = N / SZ;
    for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
      let a = 0, n = 0;
      for (let v = Math.floor(y * f); v < Math.ceil((y + 1) * f); v++) for (let u = Math.floor(x * f); u < Math.ceil((x + 1) * f); u++) { a += Math.max(-1, Math.min(1, src[v * N + u])); n++; }
      out[o + y * SZ + x] = Math.round(Math.max(0, Math.min(255, (a / n + bias) * scale)));
    }
  };
  let seed = 12345 + si;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const X = [], S = [], Y = [], meta = [];
  let skipped = 0, n = 0;
  const t0 = Date.now();
  for (const t of truth) {
    const raw = await readImage(path.join(folder, t.file));
    const img = im.fromRGBA(raw.data, raw.width, raw.height);
    let d = null;
    try { d = rec.analyzeDiagram(img); } catch (e) { console.log('ERROR', t.file, e.message); }
    // only diagrams whose board was found where it is (else the pieces would be learned on the wrong squares)
    // without a known rectangle (a downloaded dataset: only the position is known) the squares the analysis finds
    // occupied must match the position on 60 of the 64 squares
    let ok = false;
    if (t.board) {
      const [bx, by, bw, bh0] = t.board, bh = bh0 ?? bw;
      ok = d?.an && Math.abs(d.board.xs[0] - bx) < bw / 32 && Math.abs(d.board.ys[0] - by) < bh / 32 && Math.abs(d.board.xs[8] - (bx + bw)) < bw / 32 && Math.abs(d.board.ys[8] - (by + bh)) < bh / 32;
    } else if (d?.an) {
      ok = d.an.squares.filter((s) => (t.grid[s.r][s.c] !== '.') === !!s.occ).length >= 60;
    }
    if (!ok) { skipped++; continue; }
    const ctx = net.diagramContext(d.an);
    for (const s of d.an.squares) {
      const label = t.grid[s.r][s.c];
      const keep = label !== '.' || s.occ || s.area > 0.01 || s.arrow || rnd() < emptyShare;
      if (!keep) continue;
      const x = new Uint8Array(2 * SZ * SZ);
      pool(s.clean, 255, 0, x, 0);
      pool(s.resid, 127.5, 1, x, SZ * SZ);
      X.push(x);
      S.push(net.squareScalars(s, ctx));
      Y.push(net.CLASSES.indexOf(label));
      meta.push([n, t.file, t.set, t.style, t.deg, s.i]);
    }
    n++;
    if (n % 200 === 0) process.stderr.write(`${n}/${truth.length} diagrams, ${Y.length} squares, ${((Date.now() - t0) / n).toFixed(0)} ms each\n`);
  }
  fs.writeFileSync(outPrefix + '.x', Buffer.concat(X.map((x) => Buffer.from(x.buffer))));
  fs.writeFileSync(outPrefix + '.s', Buffer.concat(S.map((s) => Buffer.from(Float32Array.from(s).buffer))));
  fs.writeFileSync(outPrefix + '.y', Buffer.from(Uint8Array.from(Y)));
  fs.writeFileSync(outPrefix + '.json', JSON.stringify({ n: Y.length, N: SZ, scalars: net.SCALARS.length, classes: net.CLASSES, meta }));
  console.log(`${n} diagrams (${skipped} skipped), ${Y.length} squares → ${outPrefix}`);
})();
