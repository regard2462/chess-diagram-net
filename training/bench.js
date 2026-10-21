// Benchmark of the chess diagram recognition (diagram/) on synthetic book diagrams, in Node.
//   python training/gen_chess.py book <folder> 320        classic book diagrams (hatching, arrows, tints …)
//   python training/gen_chess.py bench <folder> [n]       the older bench in many board styles
//   node training/bench.js <folder> [--templates <file>] [--only style=hatch45,deg=dim] [--limit n] [--loo]
//                                      [--fixtures <dir>] [--json <file>] [--lib <recogniser folder>] [--verbose] [--misses]
// The templates are those of a host app that renders local piece sets (templates.js); without them the network decides alone;
// --loo leaves the piece set of each diagram out of the templates (accuracy on piece sets the recogniser has never
// seen). --fixtures adds fixture diagrams from a folder (fixtures.json: file → FEN; drawn by
// python training/gen_chess.py fixtures in the style of a printed book).
// Square accuracy is measured in image space (the diagram as printed), the orientation and the full FEN separately.
// Calibration: the confidence the card shows against the share of diagrams read without any error, per band.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { readImage } = require('./lib/image-file');

const args = process.argv.slice(2);
const folder = args[0] && !args[0].startsWith('--') ? args[0] : null;
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const flag = (name) => args.includes(name);
if (!folder && !opt('--fixtures', '')) { console.log('usage: node training/bench.js [<folder>] [--templates file] [--only k=v,…] [--limit n] [--loo] [--fixtures dir] [--json file] [--verbose]'); process.exit(1); }

const expand = (placement) => placement.split('/').map((r) => r.replace(/\d/g, (d) => '.'.repeat(+d)));
/** image rows of a FEN placement seen from `orientation` */
const gridOf = (fen, orientation) => { const g = expand(fen.split(' ')[0]); return orientation === 'black' ? g.map((r) => [...r].reverse().join('')).reverse() : g; };

(async () => {
  // --lib <folder>: another copy of the recogniser (an older version, for a comparison on the same diagrams)
  const libDir = opt('--lib', path.join(__dirname, '..', 'diagram'));
  const lib = await import(pathToFileURL(path.join(libDir, 'recognize.js')).href);
  const img = await import(pathToFileURL(path.join(libDir, 'image.js')).href);
  let truth = folder && fs.existsSync(path.join(folder, 'truth.json')) ? JSON.parse(fs.readFileSync(path.join(folder, 'truth.json'), 'utf8')) : [];
  truth = truth.map((t) => ({ ...t, path: path.join(folder, t.file) }));
  const only = (opt('--only', '') || '').split(',').filter(Boolean).map((kv) => kv.split('='));
  truth = truth.filter((t) => only.every(([k, v]) => String(t[k]) === v));
  const limit = +opt('--limit', 0); if (limit) truth = truth.slice(0, limit);
  // fixture diagrams: fixtures.json { file: { fen, orientation } } next to the pictures
  const fixDir = opt('--fixtures', '');
  if (fixDir) {
    const list = JSON.parse(fs.readFileSync(path.join(fixDir, 'fixtures.json'), 'utf8'));
    for (const [file, f] of Object.entries(list)) truth.push({ file, path: path.join(fixDir, file), set: 'real', style: f.style || 'real', deg: 'real', kind: 'fixture', orientation: f.orientation || 'white', fen: f.fen, grid: gridOf(f.fen, f.orientation || 'white'), fixture: true });
  }
  const tfile = opt('--templates', folder ? path.join(folder, '..', 'templates.json') : '');
  const dump = tfile && fs.existsSync(tfile) ? JSON.parse(fs.readFileSync(tfile, 'utf8')) : null;
  if (!dump) console.log('no templates file' + (tfile ? ' (' + tfile + ')' : '') + ': the network alone');
  const t0 = Date.now();
  const base = dump ? lib.buildTemplates(dump.boards.map((b) => ({ ...b, img: { w: b.w, h: b.h, d: Float32Array.from(Buffer.from(b.d, 'base64'), (v) => v / 255) } }))) : null;
  if (base) console.log(`templates: ${base.items.length} from ${dump.boards.length} boards in ${Date.now() - t0} ms (sets: ${[...new Set(dump.boards.map((b) => b.set))].join(', ')})`);
  const stats = new Map();
  const add = (key, field, v) => { const s = stats.get(key) || { n: 0, sq: 0, sqOk: 0, loc: 0, orient: 0, fen: 0, board: 0, ms: 0 }; s[field] += v; stats.set(key, s); };
  let done = 0;
  const errKinds = {}, roleConf = {}, lineErr = [], errByStyle = {};
  const calib = [], sqCalib = [], out = [];
  for (const t of truth) {
    const raw = await readImage(t.path);
    const gimg = img.fromRGBA(raw.data, raw.width, raw.height);
    const ts = Date.now();
    let res = null;
    try {
      // leave-one-out: the diagram's piece set (DejaVu Sans with its bold cut) is not among the templates
      const family = /^dejavu/.test(t.set) ? ['dejavu', 'dejavu-bold'] : [t.set];
      const templates = base && flag('--loo') ? base.without(...family) : base;
      res = lib.recognize(gimg, { templates });
    } catch (e) { console.log('ERROR', t.file, e.stack); }
    const ms = Date.now() - ts;
    // localisation: the found board against the truth rectangle (pre-rotation coordinates); fixtures have none
    let locOk = !!res?.board, sqOk = 0;
    if (res?.board && t.board) {
      const [bx, by, bw, bh0] = t.board, bh = bh0 ?? bw;
      const p = bw / 8, q = bh / 8;
      const { xs, ys } = res.board;
      locOk = Math.abs(xs[0] - bx) < p * 0.25 && Math.abs(ys[0] - by) < q * 0.25 && Math.abs(xs[8] - (bx + bw)) < p * 0.25 && Math.abs(ys[8] - (by + bh)) < q * 0.25;
      if (locOk && !t.rot) {
        let e = 0; for (let k = 0; k <= 8; k++) e += Math.abs(xs[k] - (bx + k * p)) / p + Math.abs(ys[k] - (by + k * q)) / q;
        lineErr.push(e / 18); (errByStyle[t.style] ||= []).push(e / 18);
      }
    }
    const errs = [];
    if (res?.grid && locOk) for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      const a = t.grid[r][c], b = res.grid[r][c];
      // '?': a square of a real book without a checked label (training/pack.py) counts as read right
      if (a === '?') { sqOk++; continue; }
      const sq = res.squares?.[r * 8 + c];
      if (sq && (a !== '.' || b !== '.')) sqCalib.push({ conf: sq.conf, ok: a === b });
      if (a === b) { sqOk++; continue; }
      errs.push(`${'abcdefgh'[c]}${8 - r}:${a}→${b}`);
      const kind = a === '.' ? 'false-piece' : b === '.' ? 'missed-piece' : (a === a.toUpperCase()) !== (b === b.toUpperCase()) ? 'colour' : 'role';
      errKinds[kind] = (errKinds[kind] || 0) + 1;
      if (kind === 'role') { const k = a.toLowerCase() + '→' + b.toLowerCase(); roleConf[k] = (roleConf[k] || 0) + 1; }
    }
    const orientOk = res?.orientation === t.orientation;
    const fenOk = !!(res?.placement && res.placement === t.fen.split(' ')[0]);
    const boardOk = locOk && sqOk === 64;
    if (res?.ok && typeof res.confidence === 'number') calib.push({ conf: res.confidence, ok: boardOk, fen: fenOk, fixture: !!t.fixture });
    out.push({ file: t.file, set: t.set, deg: t.deg, style: t.style, kind: t.kind, arrows: t.arrows || 0, located: locOk, squares: sqOk, board: boardOk, fen: fenOk, confidence: res?.confidence ?? null, uncertain: res?.uncertain?.length ?? null, errors: errs, ms });
    const keys = ['all', `style:${t.style}`, `deg:${t.deg}`, `set:${/^proc-/.test(t.set) ? 'proc' : t.set}`];
    if (t.kind) keys.push(`kind:${t.kind}`);
    if (t.arrows != null) keys.push(t.arrows ? 'arrows:yes' : 'arrows:no');
    for (const key of keys) {
      add(key, 'n', 1); add(key, 'loc', locOk ? 1 : 0); add(key, 'sq', 64); add(key, 'sqOk', sqOk); add(key, 'orient', orientOk ? 1 : 0); add(key, 'fen', fenOk ? 1 : 0); add(key, 'ms', ms);
      add(key, 'board', boardOk ? 1 : 0);
    }
    if (t.fixture) console.log(`${boardOk ? 'PASS' : 'FAIL'} fixture-${t.file.replace(/\.png$/, '')} ${sqOk}/64 squares, ${Math.round((res?.confidence || 0) * 100)} % sure, ${res?.uncertain?.length ?? '-'} unsure ${errs.join(' ')}`);
    if (flag('--verbose') || (t.fixture) || (!locOk && flag('--misses'))) {
      console.log(`${t.file} ${t.set}/${t.style}/${t.deg} loc=${locOk ? 'ok' : 'MISS'} sq=${sqOk}/64 orient=${res?.orientation}/${t.orientation} conf=${res?.confidence?.toFixed?.(3)} ${ms}ms ${errs.slice(0, 8).join(' ')}${res?.board ? '' : ' (no board)'}`);
    }
    if (++done % 50 === 0) process.stderr.write(`${done}/${truth.length}\n`);
  }
  lineErr.sort((a, b) => a - b);
  if (lineErr.length) console.log(`line error (share of a square): mean ${(lineErr.reduce((a, b) => a + b, 0) / lineErr.length).toFixed(3)}, 90 % ${lineErr[Math.floor(lineErr.length * 0.9)].toFixed(3)}, max ${lineErr[lineErr.length - 1].toFixed(3)}`);
  console.log('line error by style:', Object.entries(errByStyle).map(([k, v]) => `${k} ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(3)}`).join(', '));
  console.log('errors by kind:', JSON.stringify(errKinds));
  console.log('role confusions:', JSON.stringify(Object.fromEntries(Object.entries(roleConf).sort((a, b) => b[1] - a[1]))));
  const rows = [...stats.entries()].sort((a, b) => (a[0] === 'all' ? -1 : b[0] === 'all' ? 1 : a[0].localeCompare(b[0])));
  console.log('\nkey                      n   located  squares   board    fen      orient   ms/diagram');
  const pc = (a, b) => (b ? ((100 * a) / b).toFixed(1).padStart(6) + ' %' : '     -  ');
  for (const [k, s] of rows) console.log(`${k.padEnd(22)} ${String(s.n).padStart(4)}  ${pc(s.loc, s.n)} ${pc(s.sqOk, s.sq)} ${pc(s.board, s.n)} ${pc(s.fen, s.n)} ${pc(s.orient, s.n)}  ${(s.ms / s.n).toFixed(0).padStart(5)}`);
  // calibration: shown confidence against diagrams read without an error
  const bands = [[0, 0.5], [0.5, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 0.95], [0.95, 1.01]];
  const table = (items, label) => {
    console.log(`\ncalibration (${label}): band        n    mean shown   right`);
    let ece = 0, brier = 0;
    for (const [lo, hi] of bands) {
      const b = items.filter((x) => x.conf >= lo && x.conf < hi);
      if (!b.length) { console.log(`  ${(lo * 100).toFixed(0).padStart(3)}–${Math.min(100, hi * 100).toFixed(0).padStart(3)} %   ${'0'.padStart(5)}`); continue; }
      const m = b.reduce((a, x) => a + x.conf, 0) / b.length, acc = b.filter((x) => x.ok).length / b.length;
      ece += (b.length / items.length) * Math.abs(m - acc);
      console.log(`  ${(lo * 100).toFixed(0).padStart(3)}–${Math.min(100, hi * 100).toFixed(0).padStart(3)} %   ${String(b.length).padStart(5)}    ${(m * 100).toFixed(1).padStart(6)} %   ${(acc * 100).toFixed(1).padStart(6)} %`);
    }
    for (const x of items) brier += (x.conf - (x.ok ? 1 : 0)) ** 2;
    console.log(`  expected calibration error ${(ece * 100).toFixed(1)} %, Brier ${(brier / Math.max(1, items.length)).toFixed(3)}`);
  };
  table(calib, 'whole diagram, all squares right');
  table(sqCalib, 'squares with a piece in the truth or the result');
  const jfile = opt('--json', '');
  if (jfile) fs.writeFileSync(jfile, JSON.stringify({ rows: Object.fromEntries(stats), diagrams: out }, null, 1));
})();
