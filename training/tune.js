// Tunes the decision of the diagram recognition (recognize.js TUNE) on generated diagrams that neither the network
// nor the benchmark has seen: every diagram is analysed once (board, squares, network), then only the decision is
// repeated for each setting (cheap). The settings are searched one at a time for the most diagrams read without an
// error; then the shown confidence is calibrated: a logistic map from the decoder's probability of its position to
// the share of right positions, and the same for single squares (yellow marks).
//   python training/gen_chess.py book <folder> 400 31337 cburnett,segoe,dejavu,dejavu-bold,gothic,proc:920-959 0.15
//   node training/tune.js <folder> --templates <dump> [--limit n] [--start '{"netT":0.7}'] [--no-search] [--out file]
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const png = require('./lib/png-lite');

const args = process.argv.slice(2);
const folder = args[0];
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };

(async () => {
  const D = path.join(__dirname, '..', 'diagram');
  const rec = await import(pathToFileURL(path.join(D, 'recognize.js')).href);
  const im = await import(pathToFileURL(path.join(D, 'image.js')).href);
  const dump = JSON.parse(fs.readFileSync(opt('--templates'), 'utf8'));
  const base = rec.buildTemplates(dump.boards.map((b) => ({ ...b, img: { w: b.w, h: b.h, d: Float32Array.from(Buffer.from(b.d, 'base64'), (v) => v / 255) } })));
  let truth = [];
  for (const f of fs.readdirSync(folder).filter((f) => /^truth(-\d+)?\.json$/.test(f)).sort()) truth.push(...JSON.parse(fs.readFileSync(path.join(folder, f), 'utf8')));
  const limit = +opt('--limit', 0); if (limit) truth = truth.slice(0, limit);
  const items = [];
  const t0 = Date.now();
  for (const t of truth) {
    const raw = png.decode(fs.readFileSync(path.join(folder, t.file)));
    const res = rec.recognize(im.fromRGBA(raw.data, raw.width, raw.height), { templates: base });
    if (!res?.state) continue;
    const [bx, by, bw, bh0] = t.board, bh = bh0 ?? bw;
    const { xs, ys } = res.board;
    if (!(Math.abs(xs[0] - bx) < bw / 32 && Math.abs(ys[0] - by) < bh / 32 && Math.abs(xs[8] - (bx + bw)) < bw / 32 && Math.abs(ys[8] - (by + bh)) < bh / 32)) continue;
    items.push({ state: res.state, grid: t.grid, kind: t.kind, set: t.set });
    if (items.length % 50 === 0) process.stderr.write(`${items.length} analysed, ${((Date.now() - t0) / items.length).toFixed(0)} ms each\n`);
  }
  console.log(`${items.length} of ${truth.length} diagrams located and analysed`);
  const run = (tune) => {
    rec.setTuning(tune);
    let boards = 0, wrongSq = 0;
    const out = [];
    for (const it of items) {
      const r = rec.reclassify(it.state, base, new Map());
      let wrong = 0;
      const sq = [];
      for (let i = 0; i < 64; i++) {
        const a = it.grid[i >> 3][i & 7], b = r.grid[i >> 3][i & 7];
        if (a !== b) wrong++;
        if (a !== '.' || b !== '.') sq.push({ p: r.squares[i].conf, ok: a === b });
      }
      if (!wrong) boards++;
      wrongSq += wrong;
      out.push({ raw: r.rawConfidence, ok: !wrong, sq, kind: it.kind, rules: r.rules.length > 0 });
    }
    return { boards, wrongSq, out };
  };
  const score = (r) => r.boards - 0.02 * r.wrongSq;
  let best = { ...rec.tuning(), ...(opt('--start', '') ? JSON.parse(opt('--start', '')) : {}) };
  let bestRun = run(best);
  console.log('start', JSON.stringify(best), `boards ${bestRun.boards}/${items.length}, wrong squares ${bestRun.wrongSq}`);
  const grid = {
    netT: [0.35, 0.5, 0.6, 0.7, 0.85, 1, 1.3],
    wTpl: [0, 0.15, 0.3, 0.5, 0.8],
    knownBonus: [0, 2, 4, 6, 9],
    countWeight: [0, 0.25, 0.5, 1, 1.5],
  };
  for (let round = 0; round < (args.includes('--no-search') ? 0 : 2); round++) {
    for (const [k, vals] of Object.entries(grid)) {
      for (const v of vals) {
        if (v === best[k]) continue;
        const t = { ...best, [k]: v };
        const r = run(t);
        if (score(r) > score(bestRun)) { best = t; bestRun = r; console.log(`  ${k} = ${v}: boards ${r.boards}, wrong squares ${r.wrongSq}`); }
      }
    }
  }
  console.log('best', JSON.stringify(best), `boards ${bestRun.boards}/${items.length}, wrong squares ${bestRun.wrongSq}`);
  // calibration: logistic regression of "right" on the logit of the raw probability (Newton's method)
  // logistic regression on features [x, 1, (z)] by damped Newton steps, with a prior that pulls the map towards
  // "as it is" (slope 1, no shift): with few wrong diagrams the data alone would push it to infinity
  const fit = (pts, k, lambda = 2) => {
    const prior = new Array(k).fill(0); prior[0] = 1;
    const w = prior.slice();
    const loss = (w) => {
      let l = 0;
      for (const { f, y } of pts) { let z = 0; for (let i = 0; i < k; i++) z += w[i] * f[i]; l += Math.log(1 + Math.exp(-Math.abs(z))) + Math.max(z, 0) - y * z; }
      for (let i = 0; i < k; i++) l += 0.5 * lambda * (w[i] - prior[i]) ** 2;
      return l;
    };
    for (let it = 0; it < 80; it++) {
      const g = new Array(k).fill(0), H = Array.from({ length: k }, () => new Array(k).fill(0));
      for (const { f, y } of pts) {
        let z = 0; for (let i = 0; i < k; i++) z += w[i] * f[i];
        const p = 1 / (1 + Math.exp(-z)), v = p * (1 - p) + 1e-9;
        for (let i = 0; i < k; i++) { g[i] += (p - y) * f[i]; for (let j = 0; j < k; j++) H[i][j] += v * f[i] * f[j]; }
      }
      for (let i = 0; i < k; i++) { H[i][i] += lambda; g[i] += lambda * (w[i] - prior[i]); }
      // solve H d = g (Gauss-Jordan)
      const A = H.map((row, i) => [...row, g[i]]);
      for (let i = 0; i < k; i++) {
        let piv = i; for (let r = i + 1; r < k; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
        [A[i], A[piv]] = [A[piv], A[i]];
        for (let r = 0; r < k; r++) if (r !== i) { const m = A[r][i] / A[i][i]; for (let c = i; c <= k; c++) A[r][c] -= m * A[i][c]; }
      }
      const d = A.map((row, i) => row[k] / row[i]);
      // halve the step until the loss goes down
      const l0 = loss(w);
      let step = 1;
      for (; step > 1e-4; step /= 2) { const t = w.map((v, i) => v - step * d[i]); if (loss(t) <= l0) { t.forEach((v, i) => { w[i] = v; }); break; } }
      if (step <= 1e-4 || Math.max(...d.map(Math.abs)) * step < 1e-7) break;
    }
    return w.map((v) => +v.toFixed(3));
  };
  const lg = (p) => { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return Math.log(p / (1 - p)); };
  rec.setTuning({ ...best, calibA: 1, calibB: 0, calibC: 0, sqA: 1, sqB: 0 });
  const plain = run(rec.tuning());
  const [ca, cb, cc] = fit(plain.out.map((o) => ({ f: [lg(o.raw), 1, o.rules ? 1 : 0], y: o.ok ? 1 : 0 })), 3);
  const [sa, sb] = fit(plain.out.flatMap((o) => o.sq.map((s) => ({ f: [lg(s.p), 1], y: s.ok ? 1 : 0 }))), 2);
  const final = { ...best, calibA: ca, calibB: cb, calibC: cc, sqA: sa, sqB: sb };
  console.log('calibration', JSON.stringify({ board: [ca, cb, cc], square: [sa, sb] }));
  console.log('TUNE', JSON.stringify(final));
  if (opt('--out', '')) fs.writeFileSync(opt('--out', ''), JSON.stringify(final));
})();
