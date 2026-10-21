# Piece sets from the books themselves: for every checked book (harvest.py, label.py apply) the clearest square of each
# of the twelve pieces on a light square is rendered again from the page at a high resolution and turned into a
# sprite (the piece on a transparent square, its body opaque), so that gen_chess.py can draw new diagrams with the
# book's own pieces (CHESS_EXT_SETS, set names 'ext-book-<id>').
#   python training/book-sprites.py <harvest folder> <books folder> <out sets folder> [<book id>,…]
#       → <out>/book-<id>/{w,b}{K,Q,R,B,N,P}.png (256 × 256 RGBA) and <out>/book-sheet.png to check them by eye
# Only books where every piece was found on a light square get a set. Needs PyMuPDF, numpy and Pillow.
import json, os, sys
import numpy as np
import fitz
from PIL import Image, ImageDraw, ImageFont

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
SIZE = 256


def fill_holes(m):
    """everything not reachable from the border through unset pixels becomes set"""
    h, w = m.shape
    out = np.ones_like(m, dtype=bool)
    seen = np.zeros_like(m, dtype=bool)
    stack = [(y, x) for x in range(w) for y in (0, h - 1)] + [(y, x) for y in range(h) for x in (0, w - 1)]
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= h or x >= w or seen[y, x] or m[y, x]: continue
        seen[y, x] = True; out[y, x] = False
        stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return out


def sprite(g, white):
    """grey square (0 = black, 1 = paper) → RGBA sprite: the ink and the inside of its outline opaque; specks of the
    paper (a screened or noisy scan) and whatever touches the square's sides are left out"""
    from scipy import ndimage
    # a screened print is dots at this size: a blur of about a dot's spacing joins them into strokes again
    g = ndimage.gaussian_filter(g, 2.5)
    paper = np.percentile(g, 90)
    ink = np.percentile(g, 1)
    m = g < paper - 0.45 * max(0.2, paper - ink)
    # close small gaps of the outline (a pixel of a thin line lost in the print) before filling it
    md = ndimage.binary_closing(m, np.ones((5, 5)), iterations=1)
    lab, n = ndimage.label(md)
    if n:
        sizes = ndimage.sum(md, lab, range(1, n + 1))
        h, w = g.shape
        keep = np.zeros(n + 1, bool)
        for k in range(1, n + 1):
            ys, xs = np.where(lab == k) if sizes[k - 1] > 0.004 * h * w else ([], [])
            if not len(ys): continue
            # a piece stands in the middle; a frame or the neighbour reaches in from a side
            cy, cx = ys.mean() / h, xs.mean() / w
            if 0.15 < cx < 0.85 and 0.1 < cy < 0.9: keep[k] = True
        md = keep[lab]
        m = m & md
    sil = fill_holes(md)
    # the piece must not touch the square's sides with much of its silhouette (else it is the neighbour or a frame)
    level = np.clip((g - ink) / max(0.15, paper - ink), 0, 1)
    rgba = np.zeros(g.shape + (4,), np.uint8)
    v = (level * 255).astype(np.uint8)
    if white: v = np.where(m, v, 255)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = v
    rgba[..., 3] = np.where(sil, 255, 0)
    return rgba, sil


def main(hdir, root, out, only=None):
    books = json.load(open(os.path.join(hdir, 'books.json'), encoding='utf8'))
    os.makedirs(out, exist_ok=True)
    rows = []
    for bid, b in sorted(books.items()):
        if only and bid not in only: continue
        d = os.path.join(hdir, bid)
        if not os.path.isdir(d): continue
        prefixes = [os.path.join(d, f[:-5]) for f in sorted(os.listdir(d)) if f.startswith('c') and f.endswith('.json') and not f.endswith('.lab.json')]
        cand = {c: [] for c in CLASSES[1:]}
        for p in prefixes:
            if not os.path.exists(p + '.lab.json'): continue
            info = json.load(open(p + '.json', encoding='utf8'))
            if not info['n']: continue
            y = np.fromfile(p + '.y', dtype=np.uint8)
            X = np.fromfile(p + '.x', dtype=np.uint8).reshape(info['n'], 2, info['N'], info['N'])
            for i, m in enumerate(info['meta']):
                if y[i] == 255 or y[i] == 0 or not m[2]: continue
                cand[CLASSES[y[i]]].append((info['diagrams'][m[0]], m[1], X[i, 0].astype(np.float32)))
        if any(len(v) < 2 for v in cand.values()): continue
        doc = fitz.open(os.path.join(root, b['file']))
        tiles, ok = {}, True
        for c, lst in cand.items():
            # the square closest to the mean picture of its piece: a typical, clean print of the glyph
            mean = np.mean([t for _, _, t in lst], 0)
            dg, sq, _ = min(lst, key=lambda e: float(((e[2] - mean) ** 2).sum()))
            r, k = sq >> 3, sq & 7
            z = dg['zoom']
            x0 = (dg['crop'][0] + dg['xs'][k]) / z; x1 = (dg['crop'][0] + dg['xs'][k + 1]) / z
            y0 = (dg['crop'][1] + dg['ys'][r]) / z; y1 = (dg['crop'][1] + dg['ys'][r + 1]) / z
            Z = SIZE / max(1e-3, x1 - x0)
            pix = doc[dg['page']].get_pixmap(matrix=fitz.Matrix(Z, Z), clip=fitz.Rect(x0, y0, x1, y1), colorspace=fitz.csGRAY, alpha=False)
            g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(np.float32) / 255
            g = np.array(Image.fromarray((g * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS), np.float32) / 255
            rgba, sil = sprite(g, c.isupper())
            # a silhouette that is no piece: nearly nothing, or most of the square (a background that did not separate)
            if sil.mean() < 0.04 or sil.mean() > 0.7: ok = False; break
            tiles[c] = rgba
        if not ok: continue
        sd = os.path.join(out, f'book-{bid}')
        os.makedirs(sd, exist_ok=True)
        for c, rgba in tiles.items():
            Image.fromarray(rgba, 'RGBA').save(os.path.join(sd, ('w' if c.isupper() else 'b') + c.upper() + '.png'))
        rows.append((bid, tiles))
        print(bid, 'set written', flush=True)
    # the sheet: per book its twelve sprites on light and grey squares
    if rows:
        cell = 64
        sheet = Image.new('RGB', (110 + 12 * cell, len(rows) * cell), 'white')
        dr = ImageDraw.Draw(sheet)
        for r, (bid, tiles) in enumerate(rows):
            dr.text((4, r * cell + 24), bid, fill=(0, 0, 0))
            for k, c in enumerate('KQRBNPkqrbnp'):
                bg = Image.new('RGBA', (cell, cell), (238, 232, 220, 255) if (r + k) % 2 == 0 else (150, 150, 150, 255))
                bg.alpha_composite(Image.fromarray(tiles[c], 'RGBA').resize((cell, cell), Image.LANCZOS))
                sheet.paste(bg.convert('RGB'), (110 + k * cell, r * cell))
        sheet.save(os.path.join(out, 'book-sheet.png'))
    print(len(rows), 'book sets')


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], a[3], set(a[4].split(',')) if len(a) > 4 else None)
