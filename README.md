# Chess Diagram Net

Reads the position from a picture of a printed chess diagram, offline and in plain JavaScript. A small convolutional
network classifies every square (empty or one of the twelve pieces), and the code around it finds the board, cleans
the squares and checks the result against the rules of chess.

It comes from the chess diagram recognition of the e-book reader Cozy Shelves (version 0.10.2) and runs without it.
[Deutsche Fassung](README.de.md)

## Accuracy

On 1309 diagrams from 23 chess books the network never saw during training, 94.9 % of the diagrams are read without
a single wrong square. The books cover hatched, grey, stippled and plain dark squares, frames, coordinates and arrows,
and scans of very different quality.

## Use

Node 18 or newer, no dependencies:

```
node example/read.mjs example/diagram.png
```

prints

```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
orientation white, confidence 98 %
```

In your own code:

```js
import { fromRGBA } from './diagram/image.js';
import { recognize } from './diagram/recognize.js';

const res = recognize(fromRGBA(rgbaBytes, width, height));
// res.placement: FEN piece placement from White's side
// res.orientation: 'white' or 'black' (which side is at the bottom of the picture)
// res.confidence: probability that the whole position is right
// res.squares[i].conf, res.uncertain: per square
```

The picture should show the diagram with a little margin. `diagram/find-boards.js` finds up to two boards on a whole
page. The recogniser runs on the CPU in under a second per diagram, also in a browser or a web worker.

## Contents

| Folder | |
| --- | --- |
| `diagram/` | the recogniser: board search (`locate.js`), square analysis (`squares.js`), arrows, the network (`net.js`, weights in `net-weights.js`, 102 KB, int8), the decision with the chess rules (`decode.js`, `recognize.js`) |
| `training/` | generator for synthetic diagrams, extraction of training squares, training, benchmark, tuning |
| `example/` | a command line example and a test picture |

## Training

The network was trained on about 890,000 squares: generated diagrams in many piece sets, board styles and
degradations, and checked squares from diagrams of real chess books. The real books are not part of this repository.
The tools for both kinds of data are in `training/`, every file explains its use at the top.

Generated data needs Python 3 with Pillow, PyMuPDF and python-chess, and the cburnett pieces of chessground
(`npm install chessground@9.2.1`). The chess glyphs of the Windows fonts are used when they are there. Training needs
PyTorch (a CPU is enough for small runs).

```
python training/gen_chess.py book data/book 2000
node training/extract.js data/book data/book-shard
python training/train.py data/book-shard --epochs 20 --out diagram/net-weights.js
python training/gen_chess.py book data/test 300
node training/bench.js data/test
```

Real books: `training/harvest.py` renders the pages of a folder of PDFs and collects the boards
(`scan-books.js`), `label.py` groups the squares by their look and writes sheets to check by hand, `pack.py` turns
the checked squares into training shards and test diagrams.

## License

AGPL-3.0-or-later, see [LICENSE](LICENSE). The cburnett chess pieces used by the generator are by Colin M.L. Burnett
(GPL-2.0-or-later) and come with chessground (GPL-3.0-or-later). The weights contain no font data.
