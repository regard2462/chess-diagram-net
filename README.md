# Chess Diagram Net

Reads the position from a picture of a printed chess diagram, offline and in plain JavaScript. A small convolutional
network classifies every square (empty or one of the twelve pieces), and the code around it finds the board, cleans
the squares and checks the result against the rules of chess.

It comes from the chess diagram recognition of the e-book reader Cozy Shelves (version 0.10.2) and runs without it.
[Deutsche Fassung](README.de.md)

## Accuracy

![Six diagrams in different printing styles and the positions read from them](docs/screenshots/styles.png)

On 1309 diagrams from 23 chess books the network never saw during training, 94.9 % of the diagrams are read without
a single wrong square. The books cover hatched, grey, stippled and plain dark squares, frames, coordinates and arrows,
and scans of very different quality.

## How to use it

### In the browser – no programming

![The demo page: a printed diagram on the left, the recognised position on the right](docs/screenshots/demo.png)

`example/demo.html` is a small page for trying it out. Drop a picture onto it, paste one with Ctrl+V or pick one of
the samples. It shows the recognised position next to the picture, marks the squares it is least sure about in
orange, and gives you the FEN to copy or a button that opens the position in the Lichess analysis board. It reads a
single diagram as well as a whole book page with up to two diagrams, and nothing leaves your computer.

The page loads the recogniser as JavaScript modules, so it has to come from a web server rather than a double-clicked
file. Any small server will do, for example with Python:

```
git clone https://github.com/regard2462/chess-diagram-net.git
cd chess-diagram-net
python -m http.server 8000
```

Then open <http://localhost:8000/example/demo.html>. Without git, use *Code → Download ZIP* on GitHub and start the
server in the unpacked folder; `npx serve` works instead of Python if you have Node.

Best results: one diagram with a little white margin, taken straight on. Scans, screenshots, JPEG pictures and
slightly skewed scans all work, see the styles above.

### On the command line

![Reading two pictures on the command line](docs/screenshots/cli.png)

Node 18 or newer, no dependencies, PNG pictures:

```
node example/read.mjs example/diagram.png
```

prints

```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
orientation white, confidence 98 %
```

### In your own code

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
page (`findBoardsInPixels`, used by the demo). The recogniser runs on the CPU in under a second per diagram, also in a
browser or a web worker. The side to move, castling rights and move number are not in a diagram; the demo adds
`w - - 0 1` and lets you switch to Black.

## Contents

| Folder | |
| --- | --- |
| `diagram/` | the recogniser: board search (`locate.js`), square analysis (`squares.js`), arrows, the network (`net.js`, weights in `net-weights.js`, 102 KB, int8), the decision with the chess rules (`decode.js`, `recognize.js`) |
| `training/` | generator for synthetic diagrams, extraction of training squares, training, benchmark, tuning |
| `example/` | the browser demo (`demo.html`, with the cburnett pieces in `pieces.js` for drawing the result), the command line example (`read.mjs`) and sample pictures in `samples/` |

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

GPL-3.0-or-later, see [LICENSE](LICENSE). The cburnett chess pieces used by the generator are by Colin M.L. Burnett
(GPL-2.0-or-later) and come with chessground (GPL-3.0-or-later); the demo draws the result with the same pieces.
The sample pictures are made by the generator. The weights contain no font data.
