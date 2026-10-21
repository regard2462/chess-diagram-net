// Reads the position of a chess diagram picture (a scan or screenshot cut around the board, PNG) and prints it as FEN.
//   node example/read.mjs <diagram.png>
import fs from 'fs';
import { createRequire } from 'module';
import { fromRGBA } from '../diagram/image.js';
import { recognize } from '../diagram/recognize.js';

const require = createRequire(import.meta.url);
const png = require('../training/lib/png-lite.js');

const file = process.argv[2];
if (!file) { console.log('usage: node example/read.mjs <diagram.png>'); process.exit(1); }
const pic = png.decode(fs.readFileSync(file));
const res = recognize(fromRGBA(pic.data, pic.width, pic.height));
if (!res.ok) { console.log('no board found' + (res.error ? ': ' + res.error : '')); process.exit(2); }
console.log(res.placement);
console.log(`orientation ${res.orientation}, confidence ${(res.confidence * 100).toFixed(0)} %` +
  (res.uncertain.length ? `, unsure squares: ${res.uncertain.length}` : ''));
