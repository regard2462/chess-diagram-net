# Synthetic chess diagrams for the diagram snip of the reader (Cozy Shelves 0.6). Nothing here comes from a real book:
# positions are taken from training/positions.txt (positions from real games) and from random games, the pieces from local sets (the
# chessground cburnett SVGs, the chess glyphs of Segoe UI Symbol, DejaVu Sans and MS Gothic, one made-up
# geometric set), the boards are drawn in typical book and screen styles.
#
#   python training/gen_chess.py bench <folder> [per-combination]
#       PNG diagrams in book styles (hatching, grey, stippled and coloured squares, frames, coordinates, captions)
#       with degradations (scan blur, noise, JPEG, black and white scans, slight rotation, small and large sizes)
#       and <folder>/truth.json with the position of every image (image-space rows, orientation, side to move)
#   python training/gen_chess.py fixtures [<folder>]
#       the pictures of build/testlib/fixtures: one diagram after 1.e4 in the style of a printed chess book (hatched dark
#       squares, a thin frame, coordinates on four sides, two arrows) as a page, a dimmed night page and a screen shot
#   python training/gen_chess.py lib <folder>
#       a test library: Schach/Diagramme im Test.pdf (image, vector and scanned diagrams with captions),
#       Schach/Unbekannte Figuren.pdf (a piece set that no template knows: for the learning check),
#       Bilder/Bildtafeln.pdf (text with a colour picture), Comics/Kurzer Comic.cbz, the EPUBs of testbooks4 if
#       given as a third argument, and <folder>/chess-truth.json (page, rectangle in PDF points, FEN, orientation)
import base64, io, json, math, os, random, re, sys, zipfile
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
FONTS = r'C:\Windows\Fonts'
ROLES = 'kqrbnp'
UNI = {'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
       'k': '\u265a', 'q': '\u265b', 'r': '\u265c', 'b': '\u265d', 'n': '\u265e', 'p': '\u265f'}
SS = 3  # supersampling of every drawing

def font(name, size, index=0):
    return ImageFont.truetype(os.path.join(FONTS, name), size, index=index)

# ------------------------------------------------------------------------------------------------ piece sets
def fill_holes(mask):
    """mask: L image, 255 = ink. Returns the silhouette: everything not reachable from the border."""
    w, h = mask.size
    m = mask.point(lambda v: 255 if v > 110 else 0)
    big = Image.new('L', (w + 2, h + 2), 0)
    big.paste(m, (1, 1))
    ImageDraw.floodfill(big, (0, 0), 128)
    out = big.crop((1, 1, w + 1, h + 1)).point(lambda v: 0 if v == 128 else 255)
    return out

_cburnett = None
def cburnett_svgs():
    global _cburnett
    if _cburnett is None:
        css = open(os.path.join(ROOT, 'node_modules', 'chessground', 'assets', 'chessground.cburnett.css'), encoding='utf8').read()
        _cburnett = {}
        for m in re.finditer(r"piece\.(\w+)\.(\w+)\s*\{\s*background-image:\s*url\('data:image/svg\+xml;base64,([^']+)'\)", css):
            role = {'pawn': 'p', 'knight': 'n', 'bishop': 'b', 'rook': 'r', 'queen': 'q', 'king': 'k'}[m.group(1)]
            _cburnett[role.upper() if m.group(2) == 'white' else role] = base64.b64decode(m.group(3))
    return _cburnett

def render_svg(svg_bytes, size):
    import fitz
    doc = fitz.open(stream=svg_bytes, filetype='svg')
    pg = doc[0]
    s = size / pg.rect.width
    pix = pg.get_pixmap(matrix=fitz.Matrix(s, s), alpha=True)
    return Image.frombytes('RGBA', (pix.width, pix.height), pix.samples).resize((size, size), Image.LANCZOS)

FONT_SETS = {
    'segoe': ('seguisym.ttf', 0),
    'dejavu': ('DejaVuSans.ttf', 0),
    'dejavu-bold': ('DejaVuSans-Bold.ttf', 0),
    'gothic': ('msgothic.ttc', 0),
}
SETS = ['cburnett', 'segoe', 'dejavu', 'dejavu-bold', 'gothic']

_piece_cache = {}
def piece_images(set_name, p):
    """12 RGBA images (p x p) of one set; white pieces have an opaque white body, like chess diagram fonts."""
    key = (set_name, p)
    if key in _piece_cache: return _piece_cache[key]
    out = {}
    big = p * SS
    if set_name.startswith('proc-'):
        # made-up sets in the style of diagram fonts (chess_pieces.py): 'proc-<seed>'
        import chess_pieces
        out = chess_pieces.render_set(chess_pieces.piece_set(int(set_name.split('-')[1])), p)
        _piece_cache[key] = out
        return out
    if set_name.startswith('ext-'):
        # a piece set of 12 sprites (wK.png … bP.png, RGBA, the piece on a transparent square, the body opaque) in
        # the folder named by CHESS_EXT_SETS: designs that are not on this computer, used for training only
        folder = os.path.join(os.environ.get('CHESS_EXT_SETS', ''), set_name[4:])
        for k in UNI:
            f = ('w' if k.isupper() else 'b') + k.upper() + '.png'
            out[k] = Image.open(os.path.join(folder, f)).convert('RGBA').resize((p, p), Image.LANCZOS)
        _piece_cache[key] = out
        return out
    if set_name == 'cburnett':
        # the lichess pieces leave parts of their bodies transparent (fine on flat squares); a diagram font has an
        # opaque body, so the silhouette is filled with white below the drawing
        for k, svg in cburnett_svgs().items():
            im = render_svg(svg, big)
            sil = fill_holes(im.split()[3])
            body = Image.new('RGBA', (big, big), (0, 0, 0, 0))
            body.paste((255, 255, 255, 255), (0, 0), sil)
            body.alpha_composite(im)
            out[k] = body.resize((p, p), Image.LANCZOS)
    elif set_name == 'geo':
        out = geo_pieces(p)
    else:
        fname, idx = FONT_SETS[set_name]
        size = int(big * 0.82)
        f = font(fname, size, idx)
        boxes = {k: f.getbbox(UNI[k]) for k in UNI}
        # the tallest piece fills 86 % of the square, all pieces stand on one base line (like diagram fonts)
        tallest = max(max(b[3] - b[1], b[2] - b[0]) for b in boxes.values())
        f = font(fname, max(4, int(size * big * 0.86 / max(1, tallest))), idx)
        boxes = {k: f.getbbox(UNI[k]) for k in UNI}
        for k in UNI:
            black = k.lower()
            b = boxes[black]
            ox = (big - (b[2] - b[0])) // 2 - b[0]
            oy = int(big * 0.93) - b[3]
            solid = Image.new('L', (big, big), 0)
            ImageDraw.Draw(solid).text((ox, oy), UNI[black], font=f, fill=255)
            sil = fill_holes(solid)
            img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
            img.paste((255, 255, 255, 255), (0, 0), sil)              # opaque body (diagram fonts have a mask glyph)
            ink = Image.new('L', (big, big), 0)
            ImageDraw.Draw(ink).text((ox, oy), UNI[k], font=f, fill=255)   # the white glyph lies on the black one
            img.paste((0, 0, 0, 255), (0, 0), ink)
            out[k] = img.resize((p, p), Image.LANCZOS)
    _piece_cache[key] = out
    return out

_scaled_cache = {}
def scaled_pieces(set_name, size):
    """the pieces of a set at any size, resized from one large rendering (fonts and SVGs are slow to render)"""
    key = (set_name, size)
    if key not in _scaled_cache:
        master = piece_images(set_name, 192)
        if len(_scaled_cache) > 400: _scaled_cache.clear()
        _scaled_cache[key] = {k: im.resize((size, size), Image.LANCZOS) for k, im in master.items()}
    return _scaled_cache[key]

def geo_pieces(p):
    """A made-up set no template knows (for the learning check): simple geometric signs."""
    big = p * SS
    out = {}
    lw = max(2, big // 16)
    for k in UNI:
        img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        white = k.isupper()
        fill = (255, 255, 255, 255) if white else (0, 0, 0, 255)
        ol = (0, 0, 0, 255)
        c = big / 2
        r = k.lower()
        if r == 'p':
            d.ellipse((c - big * .18, c - big * .05, c + big * .18, c + big * .31), fill=fill, outline=ol, width=lw)
        elif r == 'r':
            d.rectangle((c - big * .28, c - big * .26, c + big * .28, c + big * .34), fill=fill, outline=ol, width=lw)
        elif r == 'b':
            d.polygon([(c, big * .12), (c - big * .3, big * .86), (c + big * .3, big * .86)], fill=fill, outline=ol)
            d.line([(c, big * .12), (c - big * .3, big * .86), (c + big * .3, big * .86), (c, big * .12)], fill=ol, width=lw)
        elif r == 'n':
            d.polygon([(c - big * .3, big * .86), (c - big * .3, big * .14), (c + big * .3, big * .14), (c + big * .3, big * .42), (c - big * .02, big * .42), (c - big * .02, big * .86)], fill=fill, outline=ol)
            d.line([(c - big * .3, big * .86), (c - big * .3, big * .14), (c + big * .3, big * .14), (c + big * .3, big * .42), (c - big * .02, big * .42), (c - big * .02, big * .86), (c - big * .3, big * .86)], fill=ol, width=lw)
        elif r == 'q':
            pts = []
            for i in range(10):
                a = -math.pi / 2 + i * math.pi / 5
                rr = big * (.40 if i % 2 == 0 else .17)
                pts.append((c + rr * math.cos(a), c + big * .04 + rr * math.sin(a)))
            d.polygon(pts, fill=fill, outline=ol)
            d.line(pts + [pts[0]], fill=ol, width=lw)
        elif r == 'k':
            d.ellipse((c - big * .36, c - big * .32, c + big * .36, c + big * .40), fill=fill, outline=ol, width=lw)
            col = (0, 0, 0, 255) if white else (255, 255, 255, 255)
            d.line([(c, c - big * .2), (c, c + big * .28)], fill=col, width=lw * 2)
            d.line([(c - big * .22, c + big * .04), (c + big * .22, c + big * .04)], fill=col, width=lw * 2)
        out[k] = img.resize((p, p), Image.LANCZOS)
    return out

# ------------------------------------------------------------------------------------------------ boards
BOARD_STYLES = {
    # name: light colour, dark: colour or pattern, frame, coordinates by default
    'hatch': dict(light=(255, 255, 255), dark='hatch45', frame='thick'),
    'hatch-back': dict(light=(255, 255, 255), dark='hatch135', frame='double'),
    'hatch-h': dict(light=(255, 255, 255), dark='hatchH', frame='thin'),
    'grey': dict(light=(255, 255, 255), dark=(178, 178, 178), frame='thin'),
    'grey-light': dict(light=(255, 255, 255), dark=(214, 214, 214), frame='thick'),
    'stipple': dict(light=(255, 255, 255), dark='stipple', frame='thick'),
    'brown': dict(light=(240, 217, 181), dark=(181, 136, 99), frame=None),
    'green': dict(light=(238, 238, 210), dark=(118, 150, 86), frame=None),
    'blue': dict(light=(222, 227, 230), dark=(140, 162, 173), frame='thin'),
}
BOOK_STYLES = ['hatch', 'hatch-back', 'hatch-h', 'grey', 'grey-light', 'stipple']
SCREEN_STYLES = ['brown', 'green', 'blue']

def dark_square(d, x0, y0, s, kind, rnd_hatch):
    """draws a patterned dark square into ImageDraw d (supersampled coordinates)"""
    spacing, width = rnd_hatch
    if kind in ('hatch45', 'hatch135'):
        n = int(2 * s / spacing) + 2
        for i in range(-n, n + 1):
            off = i * spacing
            if kind == 'hatch45':
                pts = [(x0 + off, y0 + s), (x0 + off + s, y0)]
            else:
                pts = [(x0 + off, y0), (x0 + off + s, y0 + s)]
            d.line(pts, fill=(0, 0, 0), width=width)
    elif kind == 'hatchH':
        y = y0 + spacing / 2
        while y < y0 + s:
            d.line([(x0, y), (x0 + s, y)], fill=(0, 0, 0), width=width)
            y += spacing
    elif kind == 'stipple':
        r = max(1, width)
        y = y0 + spacing / 2; row = 0
        while y < y0 + s:
            x = x0 + spacing / 2 + (spacing / 2 if row % 2 else 0)
            while x < x0 + s:
                d.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))
                x += spacing
            y += spacing; row += 1

def grid_from_fen(fen, orientation):
    """8 strings of 8 chars ('.' = empty) in image order (row 0 = top)"""
    rows = []
    for r in fen.split()[0].split('/'):
        s = ''
        for ch in r: s += '.' * int(ch) if ch.isdigit() else ch
        rows.append(s)
    if orientation == 'black':
        rows = [row[::-1] for row in rows[::-1]]
    return rows

def draw_diagram(grid, set_name, p, style, orientation='white', coords=False, frame=None, stm_mark=None, rnd=None, coord_font='arial.ttf'):
    """Renders the board alone (plus frame and coordinates). Returns (RGB image, board rect (x, y, w, h))."""
    rnd = rnd or random.Random(1)
    st = BOARD_STYLES[style]
    frame = st['frame'] if frame is None else (None if frame == 'none' else frame)
    big = p * SS
    pad = int(big * (0.55 if coords else 0.12)) + (int(big * 0.18) if frame else 0)
    W = 8 * big + 2 * pad
    img = Image.new('RGB', (W + (int(big * 0.5) if stm_mark else 0), W), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0 = y0 = pad
    spacing = rnd.uniform(big / 7.5, big / 4.8)
    # lines and dots belong to the square glyph of the diagram font: they grow with the diagram
    width = max(SS, int(round(rnd.uniform(0.7, 1.3) * SS * max(1, p / 40))))
    for r in range(8):
        for c in range(8):
            X, Y = x0 + c * big, y0 + r * big
            if (r + c) % 2 == 0:
                d.rectangle((X, Y, X + big - 1, Y + big - 1), fill=st['light'])
            else:
                dk = st['dark']
                if isinstance(dk, tuple):
                    d.rectangle((X, Y, X + big - 1, Y + big - 1), fill=dk)
                else:
                    tile = Image.new('RGB', (big, big), st['light'])
                    dark_square(ImageDraw.Draw(tile), 0, 0, big, dk, (spacing, width))
                    img.paste(tile, (X, Y))
    if frame:
        fw = {'thin': max(SS, SS), 'thick': 3 * SS, 'double': 3 * SS}[frame]
        gap = 0 if frame == 'thin' else int(rnd.choice([0, 0, 1, 2]) * SS)
        d.rectangle((x0 - gap - fw, y0 - gap - fw, x0 + 8 * big + gap + fw - 1, y0 + 8 * big + gap + fw - 1), outline=(0, 0, 0), width=fw)
        if frame == 'double':
            g2 = gap + fw + 2 * SS
            d.rectangle((x0 - g2 - SS, y0 - g2 - SS, x0 + 8 * big + g2 + SS - 1, y0 + 8 * big + g2 + SS - 1), outline=(0, 0, 0), width=SS)
    pieces = piece_images(set_name, big // SS)
    for r in range(8):
        for c in range(8):
            ch = grid[r][c]
            if ch == '.': continue
            pim = pieces[ch].resize((big, big), Image.LANCZOS)
            img.paste(pim, (x0 + c * big, y0 + r * big), pim)
    if coords:
        f = font(coord_font, int(big * 0.32))
        files = 'abcdefgh' if orientation == 'white' else 'hgfedcba'
        ranks = '87654321' if orientation == 'white' else '12345678'
        for i in range(8):
            t = files[i]; b = f.getbbox(t)
            d.text((x0 + i * big + big / 2 - (b[2] + b[0]) / 2, y0 + 8 * big + big * 0.12 + (int(big * 0.18) if frame else 0)), t, font=f, fill=(0, 0, 0))
            t = ranks[i]; b = f.getbbox(t)
            d.text((x0 - big * 0.2 - (int(big * 0.18) if frame else 0) - b[2], y0 + i * big + big / 2 - (b[3] + b[1]) / 2), t, font=f, fill=(0, 0, 0))
    if stm_mark:
        s = int(big * 0.28)
        X = x0 + 8 * big + int(big * 0.2) + (int(big * 0.18) if frame else 0)
        Y = y0 + (8 * big - s - int(big * 0.1) if stm_mark == 'white' else int(big * 0.1))
        d.rectangle((X, Y, X + s, Y + s), fill=(255, 255, 255) if stm_mark == 'white' else (0, 0, 0), outline=(0, 0, 0), width=SS)
    out = img.resize((img.width // SS, img.height // SS), Image.LANCZOS)
    return out, (pad / SS, pad / SS, 8 * p, 8 * p)

WORDS = ('der die das und mit Schach Zug Stellung Läufer Turm Springer König Dame Bauer Endspiel Mittelspiel Angriff '
         'Verteidigung Opfer Matt Weiß Schwarz gewinnt hält remis nach dem Tausch auf droht stark schwach Linie Diagonale '
         'Feld Plan Idee Variante Partie Analyse Kombination').split()

def text_block(width, lines, size, rnd, fnt='times.ttf'):
    f = font(fnt, size)
    img = Image.new('RGB', (width, int(lines * size * 1.45) + 4), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(lines):
        x = 0; words = []
        while True:
            w = rnd.choice(WORDS)
            tw = f.getlength(' '.join(words + [w]))
            if tw > width * rnd.uniform(0.85, 1.0): break
            words.append(w)
        d.text((0, i * size * 1.45), ' '.join(words), font=f, fill=(20, 20, 20))
    return img

def compose_snip(board_img, rect, rnd, caption=None, text_above=False, text_below=False):
    """Puts the diagram on a page-like snip with margins and optional text around it."""
    bw, bh = board_img.size
    mx = [int(bw * rnd.uniform(0.02, 0.16)) for _ in range(2)]
    my = [int(bh * rnd.uniform(0.02, 0.14)) for _ in range(2)]
    parts = []
    size = max(10, int(rect[2] / 22))
    if text_above: parts.append(text_block(bw, rnd.randint(1, 2), size, rnd))
    parts.append(board_img)
    cap = None
    if caption:
        f = font(rnd.choice(['times.ttf', 'georgia.ttf', 'arial.ttf']), size)
        cap = Image.new('RGB', (bw, int(size * 1.7)), (255, 255, 255))
        d = ImageDraw.Draw(cap); tl = f.getlength(caption)
        d.text(((bw - tl) / 2, size * 0.3), caption, font=f, fill=(0, 0, 0))
        parts.append(cap)
    if text_below: parts.append(text_block(bw, rnd.randint(1, 2), size, rnd))
    H = sum(pp.height for pp in parts) + my[0] + my[1]
    W = bw + mx[0] + mx[1]
    out = Image.new('RGB', (W, H), (255, 255, 255))
    y = my[0]; brect = None
    for pp in parts:
        out.paste(pp, (mx[0], y))
        if pp is board_img: brect = (mx[0] + rect[0], y + rect[1], rect[2], rect[3])
        y += pp.height
    return out, brect

# ------------------------------------------------------------------------------------------------ degradations
def degrade(img, kind, rnd):
    """Returns (image, rotation in degrees, scale factor applied)."""
    rot = 0.0; scale = 1.0
    if kind in ('clean', 'large'):
        return img, rot, scale
    if kind == 'small':
        target = rnd.randint(150, 200)        # board pixels ~ image width
        scale = target / img.width
        img = img.resize((max(20, int(img.width * scale)), max(20, int(img.height * scale))), Image.LANCZOS)
        return img, rot, scale
    if kind == 'jpeg':
        q = rnd.randint(22, 50)
        buf = io.BytesIO(); img.save(buf, 'JPEG', quality=q); buf.seek(0)
        return Image.open(buf).convert('RGB'), rot, scale
    if kind in ('scan', 'rot', 'bw'):
        g = img.convert('L')
        if kind in ('scan', 'rot'):
            rot = rnd.uniform(-1.6, 1.6) if kind == 'rot' else rnd.uniform(-0.5, 0.5)
        if abs(rot) > 0.01:
            g = g.rotate(rot, resample=Image.BICUBIC, expand=False, fillcolor=255)
        g = g.filter(ImageFilter.GaussianBlur(rnd.uniform(0.5, 1.1)))
        if kind == 'bw':
            # a black and white scan: ink stays black, paper white, grey squares turn into a dot pattern
            # (error diffusion around the threshold, as a 1-bit scan of a printed tint does)
            t = rnd.randint(140, 175)
            g = g.point(lambda v: max(0, min(255, (v - t + 64) * 2))).convert('1')
            return g.convert('RGB'), rot, scale
        # paper tone, contrast loss, noise
        import numpy as np
        lo, hi = rnd.randint(18, 45), rnd.randint(228, 246)
        a = np.asarray(g, dtype=np.float32)
        a = lo + (hi - lo) * a / 255
        a += np.random.default_rng(rnd.randint(0, 1 << 30)).normal(0, rnd.uniform(6, 14), a.shape)
        g = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'L')
        tint = (rnd.randint(244, 255), rnd.randint(236, 248), rnd.randint(215, 236))
        rgb = Image.merge('RGB', [g.point(lambda v, t=t: v * t // 255) for t in tint])
        buf = io.BytesIO(); rgb.save(buf, 'JPEG', quality=rnd.randint(60, 85)); buf.seek(0)
        return Image.open(buf).convert('RGB'), rot, scale
    raise ValueError(kind)

# ------------------------------------------------------------------------------------------------ positions
def positions(n, rnd):
    import chess
    out = []
    src = open(os.path.join(HERE, 'positions.txt'), encoding='utf8').read()
    puzzles = [l.strip() for l in src.splitlines() if l.strip() and not l.startswith('#')]
    rnd.shuffle(puzzles)
    for i in range(n):
        kind = i % 5
        if kind in (0, 1, 2) and puzzles:
            out.append(puzzles.pop())
        elif kind == 3:
            b = chess.Board()
            for _ in range(rnd.randint(8, 70)):
                moves = list(b.legal_moves)
                if not moves: break
                caps = [m for m in moves if b.is_capture(m)]
                b.push(rnd.choice(caps) if caps and rnd.random() < 0.45 else rnd.choice(moves))
            out.append(b.fen())
        else:
            # sparse endgame: kings plus a few pieces
            b = chess.Board(None)
            sq = rnd.sample(range(64), 9)
            b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
            for s in sq[2:2 + rnd.randint(1, 6)]:
                pt = rnd.choice([chess.PAWN, chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN])
                if pt == chess.PAWN and chess.square_rank(s) in (0, 7): pt = chess.ROOK
                b.set_piece_at(s, chess.Piece(pt, rnd.random() < 0.5))
            b.turn = rnd.random() < 0.6
            out.append(b.fen())
    return out

CAPTIONS_W = ['Weiß am Zug', 'Weiß zieht und gewinnt', 'White to move', 'Diagramm {n}']
CAPTIONS_B = ['Schwarz am Zug', 'Schwarz zieht', 'Black to play', 'Diagramm {n}']

# ------------------------------------------------------------------------------------------------ bench
def bench(folder, per):
    os.makedirs(folder, exist_ok=True)
    rnd = random.Random(20261003)
    combos = []
    for s in SETS:
        for style in BOOK_STYLES + SCREEN_STYLES:
            combos.append((s, style))
    degs = ['clean', 'clean', 'clean', 'scan', 'jpeg', 'small', 'rot', 'bw', 'large']
    fens = positions(len(combos) * per * len(degs) // 3 + 50, rnd)
    truth = []
    n = 0
    for (s, style) in combos:
        for rep in range(per):
            for deg in degs:
                if deg == 'bw' and style in SCREEN_STYLES: continue
                fen = rnd.choice(fens)
                orientation = 'black' if rnd.random() < 0.2 else 'white'
                p = {'large': rnd.randint(72, 96), 'small': rnd.randint(40, 56)}.get(deg, rnd.randint(30, 66))
                coords = style in BOOK_STYLES and rnd.random() < 0.55 or orientation == 'black' and rnd.random() < 0.7
                stm = 'b' if ' b ' in fen else 'w'
                grid = grid_from_fen(fen, orientation)
                img, rect = draw_diagram(grid, s, p, style, orientation, coords=coords,
                                         stm_mark=('white' if stm == 'w' else 'black') if rnd.random() < 0.15 else None, rnd=rnd,
                                         coord_font=rnd.choice(['arial.ttf', 'times.ttf', 'georgia.ttf', 'DejaVuSans.ttf']))
                cap = None
                if rnd.random() < 0.5:
                    cap = rnd.choice(CAPTIONS_W if stm == 'w' else CAPTIONS_B).format(n=rnd.randint(1, 240))
                snip, brect = compose_snip(img, rect, rnd, caption=cap, text_above=rnd.random() < 0.3, text_below=rnd.random() < 0.25)
                snip, rot, scale = degrade(snip, deg, rnd)
                name = f'd{n:04d}.png'
                snip.save(os.path.join(folder, name))
                truth.append({'file': name, 'set': s, 'style': style, 'deg': deg, 'orientation': orientation, 'stm': stm,
                              'fen': fen, 'grid': grid, 'coords': coords, 'p': p, 'rot': round(rot, 2),
                              'board': [round(v * scale, 2) for v in brect]})
                n += 1
    json.dump(truth, open(os.path.join(folder, 'truth.json'), 'w', encoding='utf8'), indent=0)
    print('bench', n, 'diagrams in', folder)

# ------------------------------------------------------------------------------------------------ test library
def pdf_lib(folder, epub_src=None):
    import fitz
    rnd = random.Random(77)
    os.makedirs(os.path.join(folder, 'Schach'), exist_ok=True)
    truth = {'pdf': 'Schach/Diagramme im Test.pdf', 'diagrams': [], 'learn': {'pdf': 'Schach/Unbekannte Figuren.pdf', 'diagrams': []}}
    W, H = 432, 648   # 6 x 9 inch book page
    doc = fitz.open()
    serif = os.path.join(FONTS, 'georgia.ttf')
    title = doc.new_page(width=W, height=H)
    title.insert_font(fontname='geo', fontfile=serif)
    title.insert_text((60, 200), 'Diagramme im Test', fontname='geo', fontsize=26)
    title.insert_text((60, 236), 'Ein Schachbuch ohne Inhalt', fontname='geo', fontsize=14)
    para = ('Die Stellung im Diagramm zeigt einen typischen Moment der Partie. Weiß hat den Läufer aktiv postiert und '
            'droht mit dem Turm auf der offenen Linie einzudringen. Schwarz muss genau verteidigen, denn nach dem '
            'Tausch der Damen steht das Endspiel für Weiß deutlich besser.')

    def page_with_text(lines_top=3):
        pg = doc.new_page(width=W, height=H)
        pg.insert_font(fontname='geo', fontfile=serif)
        rect = fitz.Rect(54, 50, W - 54, 50 + lines_top * 16)
        pg.insert_textbox(rect, para.split('. ')[0] + '. ' + para.split('. ')[1] + '.', fontname='geo', fontsize=10.5, lineheight=1.35)
        return pg

    fens = positions(40, random.Random(5))
    specs = [
        # kind, set, style, orientation, caption side to move, coords
        ('image', 'dejavu', 'hatch', 'white', 'w', True),
        ('image', 'segoe', 'grey', 'white', 'b', False),
        ('vector', 'dejavu', 'hatch', 'white', 'w', True),
        ('vector-svg', 'cburnett', 'brown', 'white', 'b', False),
        ('image', 'cburnett', 'stipple', 'black', 'w', True),
        ('scan', 'gothic', 'hatch-h', 'white', 'b', True),
        ('image', 'dejavu-bold', 'grey-light', 'white', 'w', False),
        ('vector', 'segoe', 'grey', 'black', 'b', True),
    ]
    for i, (kind, s, style, orientation, stm, coords) in enumerate(specs):
        fen = fens[i]
        parts = fen.split()
        parts[1] = stm
        fen = ' '.join(parts)
        pg = page_with_text()
        grid = grid_from_fen(fen, orientation)
        bx, by, bs = 96, 130, 240   # board rectangle in points (without frame/coordinates)
        cap = ('Weiß am Zug' if stm == 'w' else 'Schwarz am Zug')
        if kind in ('image', 'scan'):
            p = 46
            img, rect = draw_diagram(grid, s, p, style, orientation, coords=coords, rnd=random.Random(i))
            sc = bs / (8 * p)
            if kind == 'scan':
                img, _, _ = degrade(img, 'scan', random.Random(i))
            buf = io.BytesIO(); img.save(buf, 'PNG')
            r = fitz.Rect(bx - rect[0] * sc, by - rect[1] * sc, bx - rect[0] * sc + img.width * sc, by - rect[1] * sc + img.height * sc)
            pg.insert_image(r, stream=buf.getvalue())
        else:
            vector_board(pg, grid, s, style, orientation, coords, bx, by, bs)
        pg.insert_text((bx + bs / 2 - 30, by + bs + 34), cap, fontname='geo', fontsize=11)
        pg.insert_textbox(fitz.Rect(54, by + bs + 50, W - 54, H - 50), para + ' ' + para, fontname='geo', fontsize=10.5, lineheight=1.35)
        pad = bs * 0.12
        truth['diagrams'].append({'page': doc.page_count - 1, 'kind': kind, 'set': s, 'style': style, 'orientation': orientation,
                                  'stm': stm, 'fen': fen, 'board': [bx, by, bs, bs],
                                  'snip': [bx - pad - (bs * 0.08 if coords else 0), by - pad, bs + 2 * pad + (bs * 0.08 if coords else 0), bs + 2 * pad + (bs * 0.06 if coords else 0)]})
    # a diagram in the style of a printed chess book (build/testlib/fixtures/hatched-e4-arrows-page.png, drawn by
    # `gen_chess.py fixtures`): hatched dark squares, a thin frame, coordinates on all four sides, arrows for 1.e4 and
    # for the knight's answer; its board is 200 points (7 cm) wide on the page
    fix = os.path.join(HERE, 'fixtures', 'hatched-e4-arrows-page.png')
    if os.path.exists(fix):
        pg = page_with_text()
        im = Image.open(fix)
        meta = json.load(open(os.path.join(HERE, 'fixtures', 'fixtures.json'), encoding='utf8'))['hatched-e4-arrows-page.png']
        fx0, fy0, fx1, fy1 = meta['board']      # the board inside the picture (pixels)
        bx, by, bs = 116, 130, 200
        sc = bs / (fx1 - fx0)
        r = fitz.Rect(bx - fx0 * sc, by - fy0 * sc, bx + (im.width - fx0) * sc, by + (im.height - fy0) * sc)
        buf = io.BytesIO(); im.save(buf, 'PNG')
        pg.insert_image(r, stream=buf.getvalue())
        pg.insert_text((bx + bs / 2 - 34, by + bs + 40), 'Schwarz am Zug', fontname='geo', fontsize=11)
        pg.insert_textbox(fitz.Rect(54, by + bs + 56, W - 54, H - 50), para, fontname='geo', fontsize=10.5, lineheight=1.35)
        pad = bs * 0.12
        truth['diagrams'].append({'page': doc.page_count - 1, 'kind': 'book-picture', 'set': 'book', 'style': 'hatch45', 'orientation': 'white',
                                  'stm': 'b', 'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', 'board': [bx, by, bs, bs],
                                  'snip': [bx - pad - bs * 0.1, by - pad - bs * 0.06, bs + 2 * pad + bs * 0.2, bs + 2 * pad + bs * 0.12], 'exact': True})
    doc.save(os.path.join(folder, truth['pdf']))

    # the learning check: one book, one unknown piece set, four diagrams
    learn = fitz.open()
    t = learn.new_page(width=W, height=H)
    t.insert_font(fontname='geo', fontfile=serif)
    t.insert_text((60, 200), 'Unbekannte Figuren', fontname='geo', fontsize=26)
    lf = positions(12, random.Random(9))
    for i in range(4):
        fen = lf[i]
        pg = learn.new_page(width=W, height=H)
        pg.insert_font(fontname='geo', fontfile=serif)
        pg.insert_textbox(fitz.Rect(54, 50, W - 54, 100), para.split('. ')[0] + '.', fontname='geo', fontsize=10.5, lineheight=1.35)
        grid = grid_from_fen(fen, 'white')
        p = 46
        img, rect = draw_diagram(grid, 'geo', p, 'grey', 'white', coords=False, rnd=random.Random(100 + i))
        bx, by, bs = 96, 130, 240
        sc = bs / (8 * p)
        buf = io.BytesIO(); img.save(buf, 'PNG')
        pg.insert_image(fitz.Rect(bx - rect[0] * sc, by - rect[1] * sc, bx - rect[0] * sc + img.width * sc, by - rect[1] * sc + img.height * sc), stream=buf.getvalue())
        stm = 'b' if ' b ' in fen else 'w'
        pg.insert_text((bx + bs / 2 - 30, by + bs + 34), 'Weiß am Zug' if stm == 'w' else 'Schwarz am Zug', fontname='geo', fontsize=11)
        pad = bs * 0.12
        truth['learn']['diagrams'].append({'page': learn.page_count - 1, 'set': 'geo', 'style': 'grey', 'orientation': 'white', 'stm': stm,
                                           'fen': fen, 'board': [bx, by, bs, bs], 'snip': [bx - pad, by - pad, bs + 2 * pad, bs + 2 * pad]})
    learn.save(os.path.join(folder, truth['learn']['pdf']))

    # a picture book page for the night theme: text and a colourful picture
    os.makedirs(os.path.join(folder, 'Bilder'), exist_ok=True)
    pic = fitz.open()
    pg = pic.new_page(width=W, height=H)
    pg.insert_font(fontname='geo', fontfile=serif)
    pg.insert_textbox(fitz.Rect(54, 50, W - 54, 200), para + ' ' + para, fontname='geo', fontsize=11, lineheight=1.35)
    photo = landscape(560, 380, random.Random(3))
    buf = io.BytesIO(); photo.save(buf, 'JPEG', quality=88)
    pg.insert_image(fitz.Rect(54, 230, W - 54, 230 + (W - 108) * 380 / 560), stream=buf.getvalue())
    pg.insert_textbox(fitz.Rect(54, 470, W - 54, H - 40), para, fontname='geo', fontsize=11, lineheight=1.35)
    pg2 = pic.new_page(width=W, height=H)
    pg2.insert_font(fontname='geo', fontfile=serif)
    pg2.insert_textbox(fitz.Rect(54, 50, W - 54, H - 50), (para + ' ') * 6, fontname='geo', fontsize=11, lineheight=1.35)
    pic.set_metadata({'title': 'Bildtafeln', 'author': 'Test'})
    pic.save(os.path.join(folder, 'Bilder', 'Bildtafeln.pdf'))
    truth['picture'] = {'pdf': 'Bilder/Bildtafeln.pdf', 'page': 0, 'image': [54, 230, W - 108, (W - 108) * 380 / 560]}

    # a short comic (CBZ) for the fixed-layout image path
    os.makedirs(os.path.join(folder, 'Comics'), exist_ok=True)
    with zipfile.ZipFile(os.path.join(folder, 'Comics', 'Kurzer Comic.cbz'), 'w') as z:
        for i in range(3):
            im = landscape(600, 860, random.Random(10 + i))
            buf = io.BytesIO(); im.save(buf, 'JPEG', quality=85)
            z.writestr(f'{i + 1:02d}.jpg', buf.getvalue())
    # a chess e-book (EPUB) with diagrams as pictures, the way chess publishers ship them
    truth['epub'] = {'file': 'Schach/Diagramme im E-Book.epub', 'diagrams': []}
    especs = [('cburnett', 'hatch', 'white', 'w', True), ('segoe', 'grey', 'white', 'b', False), ('dejavu', 'stipple', 'black', 'w', True)]
    efens = positions(12, random.Random(21))
    chapters, images = [], []
    for i, (s, style, orientation, stm, coords) in enumerate(especs):
        parts = efens[i].split(); parts[1] = stm; fen = ' '.join(parts)
        img, _ = draw_diagram(grid_from_fen(fen, orientation), s, 44, style, orientation, coords=coords, rnd=random.Random(300 + i))
        buf = io.BytesIO(); img.save(buf, 'PNG')
        images.append((f'img/d{i + 1}.png', buf.getvalue()))
        cap = 'Weiß am Zug' if stm == 'w' else 'Schwarz am Zug'
        chapters.append((f'c{i + 1}.xhtml', f'Partie {i + 1}',
                         f'<h2>Partie {i + 1}</h2><p>{para.split(". ")[0]}.</p>'
                         f'<div class="dia"><img src="img/d{i + 1}.png" alt="Diagramm {i + 1}"/><p class="cap">{cap}</p></div>'
                         f'<p>{para}</p><p>{para}</p>'))
        truth['epub']['diagrams'].append({'chapter': i + 1, 'set': s, 'style': style, 'orientation': orientation, 'stm': stm, 'fen': fen})
    epub_path = os.path.join(folder, truth['epub']['file'])
    with zipfile.ZipFile(epub_path, 'w') as z:
        z.writestr(zipfile.ZipInfo('mimetype'), 'application/epub+zip')
        z.writestr('META-INF/container.xml', '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        xhtml = lambda title, body: ('<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml" '
                                     f'xmlns:epub="http://www.idpf.org/2007/ops" lang="de"><head><title>{title}</title>'
                                     f'<link rel="stylesheet" href="style.css"/></head><body>{body}</body></html>')
        z.writestr('OEBPS/style.css', 'body{font-family:serif}.dia{text-align:center;margin:1em 0}.dia img{width:22em;max-width:100%}.cap{font-style:italic}')
        z.writestr('OEBPS/nav.xhtml', xhtml('Inhalt', '<nav epub:type="toc"><ol>' + ''.join(f'<li><a href="{f}">{t}</a></li>' for f, t, _ in chapters) + '</ol></nav>'))
        for f, t, body in chapters: z.writestr('OEBPS/' + f, xhtml(t, body))
        for f, data in images: z.writestr('OEBPS/' + f, data)
        z.writestr('OEBPS/content.opf', '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">lesesaal-test-diagramme-ebook</dc:identifier>'
                   '<dc:title>Diagramme im E-Book</dc:title><dc:creator>Test</dc:creator><dc:language>de</dc:language>'
                   '<meta property="dcterms:modified">2026-10-03T00:00:00Z</meta></metadata><manifest>'
                   '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="css" href="style.css" media-type="text/css"/>'
                   + ''.join(f'<item id="c{k + 1}" href="{f}" media-type="application/xhtml+xml"/>' for k, (f, _, _) in enumerate(chapters))
                   + ''.join(f'<item id="i{k + 1}" href="{f}" media-type="image/png"/>' for k, (f, _) in enumerate(images))
                   + '</manifest><spine>' + ''.join(f'<itemref idref="c{k + 1}"/>' for k in range(len(chapters))) + '</spine></package>')
    if epub_src and os.path.isdir(epub_src):
        import shutil
        os.makedirs(os.path.join(folder, 'Romane'), exist_ok=True)
        for f in [x for x in sorted(os.listdir(epub_src)) if x.endswith('.epub')][:2]:
            shutil.copy(os.path.join(epub_src, f), os.path.join(folder, 'Romane', f))
    json.dump(truth, open(os.path.join(folder, 'chess-truth.json'), 'w', encoding='utf8'), indent=1)
    print('library in', folder)

def vector_board(pg, grid, set_name, style, orientation, coords, bx, by, bs):
    """A diagram drawn with PDF vector graphics: squares, hatching, frame, pieces as font glyphs or SVG."""
    import fitz
    st = BOARD_STYLES[style]
    q = bs / 8
    shape = pg.new_shape()
    for r in range(8):
        for c in range(8):
            rc = fitz.Rect(bx + c * q, by + r * q, bx + (c + 1) * q, by + (r + 1) * q)
            if (r + c) % 2 == 0 and st['light'] != (255, 255, 255):
                shape.draw_rect(rc); shape.finish(fill=[v / 255 for v in st['light']], color=None, width=0)
            if (r + c) % 2 == 1:
                dk = st['dark']
                if isinstance(dk, tuple):
                    shape.draw_rect(rc); shape.finish(fill=[v / 255 for v in dk], color=None, width=0)
                else:
                    sp = q / 6
                    k = -8
                    while k <= 8:
                        x1, y1, x2, y2 = rc.x0 + k * sp, rc.y1, rc.x0 + k * sp + q, rc.y0
                        # clip the hatch line to the square
                        t0, t1 = 0.0, 1.0
                        for (p0, dd, lo, hi) in ((x1, x2 - x1, rc.x0, rc.x1), (y1, y2 - y1, rc.y0, rc.y1)):
                            if dd == 0: continue
                            a, b = (lo - p0) / dd, (hi - p0) / dd
                            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
                        if t1 > t0:
                            shape.draw_line((x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0), (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1))
                        k += 1
                    shape.finish(color=(0, 0, 0), width=0.45)
    shape.draw_rect(fitz.Rect(bx - 1.2, by - 1.2, bx + bs + 1.2, by + bs + 1.2))
    shape.finish(color=(0, 0, 0), width=1.6)
    shape.commit()
    if set_name == 'cburnett':
        svgs = cburnett_svgs()
        cache = {}
        for r in range(8):
            for c in range(8):
                ch = grid[r][c]
                if ch == '.': continue
                if ch not in cache: cache[ch] = fitz.open('pdf', fitz.open(stream=svgs[ch], filetype='svg').convert_to_pdf())
                pg.show_pdf_page(fitz.Rect(bx + c * q, by + r * q, bx + (c + 1) * q, by + (r + 1) * q), cache[ch], 0)
    else:
        fname, idx = FONT_SETS[set_name]
        fpath = os.path.join(FONTS, fname)
        pg.insert_font(fontname='chessf', fontfile=fpath)
        f = fitz.Font(fontfile=fpath)
        size = q * 0.86
        for r in range(8):
            for c in range(8):
                ch = grid[r][c]
                if ch == '.': continue
                x = bx + c * q + (q - f.text_length(UNI[ch], fontsize=size)) / 2
                y = by + (r + 1) * q - q * 0.14
                if ch.isupper():
                    pg.insert_text((x, y), UNI[ch.lower()], fontname='chessf', fontsize=size, color=(1, 1, 1))
                pg.insert_text((x, y), UNI[ch], fontname='chessf', fontsize=size, color=(0, 0, 0))
    if coords:
        pg.insert_font(fontname='coordf', fontfile=os.path.join(FONTS, 'arial.ttf'))
        files = 'abcdefgh' if orientation == 'white' else 'hgfedcba'
        ranks = '87654321' if orientation == 'white' else '12345678'
        for i in range(8):
            pg.insert_text((bx + i * q + q / 2 - 2.5, by + bs + 11), files[i], fontname='coordf', fontsize=8)
            pg.insert_text((bx - 10, by + i * q + q / 2 + 3), ranks[i], fontname='coordf', fontsize=8)

def landscape(w, h, rnd):
    """A colourful made-up picture (sky, sun, hills, trees, lake)."""
    img = Image.new('RGB', (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(90 + 120 * t), int(150 + 70 * t), int(235 - 40 * t)))
    d.ellipse((w * 0.7, h * 0.08, w * 0.82, h * 0.08 + w * 0.12), fill=(255, 214, 90))
    for k, col in enumerate([(70, 140, 70), (40, 110, 55), (30, 85, 45)]):
        pts = [(0, h)]
        base = h * (0.5 + 0.12 * k)
        for x in range(0, w + 20, 20):
            pts.append((x, base - 40 * math.sin(x / (60 + 30 * k) + k) - rnd.uniform(0, 12)))
        pts.append((w, h))
        d.polygon(pts, fill=col)
    d.ellipse((w * 0.1, h * 0.78, w * 0.55, h * 0.95), fill=(60, 120, 200))
    for i in range(9):
        x = rnd.uniform(0, w); y = rnd.uniform(h * 0.55, h * 0.8)
        d.rectangle((x - 3, y, x + 3, y + 26), fill=(110, 70, 40))
        d.ellipse((x - 16, y - 30, x + 16, y + 6), fill=(rnd.randint(20, 60), rnd.randint(120, 170), rnd.randint(40, 80)))
    d.rectangle((w * 0.62, h * 0.66, w * 0.74, h * 0.8), fill=(200, 60, 50))
    d.polygon([(w * 0.6, h * 0.66), (w * 0.68, h * 0.58), (w * 0.76, h * 0.66)], fill=(120, 40, 30))
    return img.filter(ImageFilter.GaussianBlur(0.6))

# ------------------------------------------------------------------------------------------------ book diagrams (0.7)
# Classic printed diagrams as chess books print them: dark squares hatched with thin lines (or grey, stippled), a thin
# frame, coordinates on two or four sides, arrows for moves drawn over the board, the piece sets of the bench plus
# made-up diagram fonts (chess_pieces.py), and what happens on the way to the snip: page tints (paper, sepia, the
# dimmed pictures of night pages), screen resolution, scans, JPEG, a picture stretched a little in one direction.
# Positions from real games (the puzzle collection), from opening lines and from games continued from them.

OPENINGS = [
    'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O',
    'e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+',
    'e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5',
    'e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6',
    'e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5',
    'e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7',
    'e4 e5 f4 exf4 Nf3 g5',
    'e4 e5 Nc3 Nf6 f4 d5',
    'e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4',
    'e4 e5 Bc4 Nf6 d3 c6',
    'e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#',
    'f3 e5 g4 Qh4#',
    'e4 e5 d4 exd4 c3 dxc3 Bc4 cxb2 Bxb2',
    'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5',
    'e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Bg7',
    'e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7',
    'e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6',
    'e4 c5 c3 d5 exd5 Qxd5 d4 Nf6',
    'e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3',
    'e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6',
    'e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6',
    'e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6',
    'e4 d6 d4 Nf6 Nc3 g6 f4 Bg7',
    'e4 Nf6 e5 Nd5 d4 d6 Nf3 g6',
    'e4 g6 d4 Bg7 Nc3 d6',
    'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6',
    'd4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5',
    'd4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5',
    'd4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5',
    'd4 Nf6 c4 e6 Nf3 b6 g3 Ba6',
    'd4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5',
    'd4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7',
    'd4 f5 g3 Nf6 Bg2 e6 Nf3 Be7',
    'd4 Nf6 c4 c5 d5 b5 cxb5 a6',
    'd4 d5 Bf4 Nf6 e3 e6 Nf3 c5 c3 Nc6',
    'c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6',
    'c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7',
    'Nf3 d5 g3 Nf6 Bg2 c6 O-O Bg4',
    'd4 d5 Nf3 Nf6 e3 e6 Bd3 c5 c3 Nc6 Nbd2 Bd6 O-O O-O',
    'e4 e5 Nf3 Nc6 Bc4 Be7 d4 d6',
]

def opening_positions():
    """every position along the opening lines (the start position included), with the move that led there"""
    import chess
    seen, out = set(), []
    b = chess.Board()
    out.append((b.fen(), None)); seen.add(b.board_fen())
    for line in OPENINGS:
        b = chess.Board()
        for san in line.split():
            mv = b.push_san(san)
            if b.board_fen() not in seen:
                seen.add(b.board_fen()); out.append((b.fen(), mv.uci()))
    return out

def book_positions(n, rnd):
    """[(fen, kind, last move or None)]: opening positions, positions a move or two away from them, positions from
    real games (puzzles), games continued from an opening, and teaching diagrams with a few pieces"""
    import chess
    src = open(os.path.join(HERE, 'positions.txt'), encoding='utf8').read()
    puzzles = [l.strip() for l in src.splitlines() if l.strip() and not l.startswith('#')]
    openings = opening_positions()
    out = []
    for i in range(n):
        u = rnd.random()
        if u < 0.32:
            fen, last = rnd.choice(openings)
            out.append((fen, 'opening', last))
        elif u < 0.44:
            b = chess.Board(rnd.choice(openings)[0])
            last = None
            for _ in range(rnd.randint(1, 2)):
                moves = list(b.legal_moves)
                if not moves: break
                mv = rnd.choice(moves); b.push(mv); last = mv.uci()
            out.append((b.fen(), 'near-opening', last))
        elif u < 0.78:
            out.append((rnd.choice(puzzles), 'game', None))
        elif u < 0.94:
            b = chess.Board(rnd.choice(openings)[0])
            last = None
            for _ in range(rnd.randint(6, 44)):
                moves = list(b.legal_moves)
                if not moves: break
                caps = [m for m in moves if b.is_capture(m)]
                mv = rnd.choice(caps) if caps and rnd.random() < 0.35 else rnd.choice(moves)
                b.push(mv); last = mv.uci()
            out.append((b.fen(), 'continued', last))
        else:
            # a teaching diagram: one piece and maybe the kings, or a small pawn ending
            b = chess.Board(None)
            sq = rnd.sample(range(64), 6)
            if rnd.random() < 0.6:
                b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
            for s in sq[2:2 + rnd.randint(1, 3)]:
                pt = rnd.choice([chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN])
                if pt == chess.PAWN and chess.square_rank(s) in (0, 7): pt = chess.KNIGHT
                b.set_piece_at(s, chess.Piece(pt, rnd.random() < 0.6))
            out.append((b.fen(), 'teaching', None))
    return out

def _sq_xy(sq, orientation):
    """python-chess square → (column, row) in the image"""
    f, r = sq % 8, sq // 8
    return (f, 7 - r) if orientation == 'white' else (7 - f, r)

def book_arrows(fen, last, rnd, orientation):
    """arrows as [(c0, r0, c1, r1, end)] in image squares: the move that led to the position (the piece stands at the
    head, so the head stops short of it) and moves the side to move might play (the head on the target square)"""
    import chess
    b = chess.Board(fen)
    arrows = []
    if last and rnd.random() < 0.7:
        mv = chess.Move.from_uci(last)
        c0, r0 = _sq_xy(mv.from_square, orientation); c1, r1 = _sq_xy(mv.to_square, orientation)
        arrows.append((c0, r0, c1, r1, rnd.uniform(0.3, 0.5)))
    try:
        moves = list(b.legal_moves)
    except Exception:
        moves = []
    if not moves:
        # a teaching diagram without kings: moves of a piece as it would go
        for s, pc in b.piece_map().items():
            if pc.piece_type in (chess.KING, chess.PAWN): continue
            bb = chess.Board(None); bb.set_piece_at(s, pc); bb.turn = pc.color
            moves += list(bb.pseudo_legal_moves)
    for _ in range(rnd.choice([0, 1, 1, 2]) if arrows else rnd.choice([1, 1, 2, 3])):
        if not moves: break
        mv = rnd.choice(moves)
        c0, r0 = _sq_xy(mv.from_square, orientation); c1, r1 = _sq_xy(mv.to_square, orientation)
        if (c0, r0, c1, r1) in [a[:4] for a in arrows]: continue
        arrows.append((c0, r0, c1, r1, rnd.choice([0.0, 0.0, 0.1, 0.2])))
    return arrows

BOOK_DARK = ['hatch45', 'hatch45', 'hatch45', 'hatch135', 'hatchH', 'grey', 'grey-light', 'stipple']
COORD_FONTS = ['arial.ttf', 'times.ttf', 'georgia.ttf', 'DejaVuSans.ttf', 'verdana.ttf', 'segoeui.ttf']
# boards of screens and apps (light and dark colour), for the training of the classifier
SCREEN_COLOURS = {'brown': ((240, 217, 181), (181, 136, 99)), 'green': ((238, 238, 210), (118, 150, 86)), 'blue': ((222, 227, 230), (140, 162, 173))}

def book_style(rnd, screen=0.0):
    if rnd.random() < screen:
        st = book_style(rnd)
        st.update({'dark': rnd.choice(list(SCREEN_COLOURS)), 'frame': rnd.choice(['none', 'none', 'thin']), 'coords': rnd.choice(['none', 'none', 'lb'])})
        return st
    return {
        'dark': rnd.choice(BOOK_DARK),
        'spacing': rnd.uniform(6.0, 11.0),           # hatch lines per square side
        'line': rnd.uniform(0.6, 1.5),               # hatch line width in pixels of the final image (at 40 px squares)
        'grey': rnd.choice([(178, 178, 178), (196, 196, 196), (160, 160, 160)]),
        'frame': rnd.choice(['thin', 'thin', 'medium', 'thick', 'double', 'none']),
        'frame_gap': rnd.choice([0, 0, 0, 1, 2]),
        'coords': rnd.choice(['lb', 'lb', 'all', 'all', 'none']),
        'coord_font': rnd.choice(COORD_FONTS),
        'coord_size': rnd.uniform(0.26, 0.4),
        'arrow_w': rnd.uniform(0.03, 0.075), 'head_len': rnd.uniform(0.18, 0.3), 'head_w': rnd.uniform(0.15, 0.28),
        'head_fill': rnd.random() < 0.7, 'arrow_grey': rnd.choice([0, 0, 0, 60]),
    }

def draw_book(grid, set_name, p, st, rnd, orientation='white', arrows=()):
    """a classic book diagram: RGB image and the board rectangle (x, y, w, h) in its pixels"""
    big = p * SS
    coords = st['coords'] != 'none'
    fw = {'thin': SS, 'medium': 2 * SS, 'thick': 3 * SS, 'double': SS, 'none': 0}[st['frame']]
    gap = st['frame_gap'] * SS if st['frame'] != 'none' else 0
    margin = int(big * (0.62 if coords else 0.12)) + fw + gap + (4 * SS if st['frame'] == 'double' else 0)
    W = 8 * big + 2 * margin
    img = Image.new('RGB', (W, W), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0 = y0 = margin
    width = max(1, int(round(st['line'] * SS * max(1.0, p / 40))))
    spacing = big / st['spacing']
    for r in range(8):
        for c in range(8):
            X, Y = x0 + c * big, y0 + r * big
            kind = st['dark']
            if kind in SCREEN_COLOURS:
                d.rectangle((X, Y, X + big - 1, Y + big - 1), fill=SCREEN_COLOURS[kind][(r + c) % 2])
                continue
            if (r + c) % 2 == 0: continue
            if kind in ('grey', 'grey-light'):
                d.rectangle((X, Y, X + big - 1, Y + big - 1), fill=st['grey'] if kind == 'grey' else (214, 214, 214))
            else:
                tile = Image.new('RGB', (big, big), (255, 255, 255))
                dark_square(ImageDraw.Draw(tile), 0, 0, big, kind, (spacing, width))
                img.paste(tile, (X, Y))
    if st['frame'] != 'none':
        o = gap + fw
        d.rectangle((x0 - o, y0 - o, x0 + 8 * big + o - 1, y0 + 8 * big + o - 1), outline=(0, 0, 0), width=fw)
        if st['frame'] == 'double':
            o2 = o + 3 * SS
            d.rectangle((x0 - o2, y0 - o2, x0 + 8 * big + o2 - 1, y0 + 8 * big + o2 - 1), outline=(0, 0, 0), width=SS)
    pieces = scaled_pieces(set_name, big)
    for r in range(8):
        for c in range(8):
            ch = grid[r][c]
            if ch == '.': continue
            pim = pieces[ch]
            img.paste(pim, (x0 + c * big, y0 + r * big), pim)
    # arrows over everything (they are drawn on top of the pieces in books)
    col = (st['arrow_grey'],) * 3
    for (c0, r0, c1, r1, end) in arrows:
        ax, ay = x0 + (c0 + 0.5) * big, y0 + (r0 + 0.5) * big
        bx, by = x0 + (c1 + 0.5) * big, y0 + (r1 + 0.5) * big
        L = math.hypot(bx - ax, by - ay)
        if L < 1: continue
        ux, uy = (bx - ax) / L, (by - ay) / L
        bx, by = bx - ux * end * big, by - uy * end * big
        ax, ay = ax + ux * 0.08 * big, ay + uy * 0.08 * big
        lw = max(SS, int(st['arrow_w'] * big))
        hl, hw = st['head_len'] * big, st['head_w'] * big
        hx, hy = bx - ux * hl, by - uy * hl
        nx, ny = -uy, ux
        if st['head_fill']:
            d.line([(ax, ay), (hx + ux * 2, hy + uy * 2)], fill=col, width=lw)
            d.polygon([(bx, by), (hx + nx * hw / 2, hy + ny * hw / 2), (hx - nx * hw / 2, hy - ny * hw / 2)], fill=col)
        else:
            d.line([(ax, ay), (bx, by)], fill=col, width=lw)
            d.line([(bx, by), (hx + nx * hw / 2, hy + ny * hw / 2)], fill=col, width=lw)
            d.line([(bx, by), (hx - nx * hw / 2, hy - ny * hw / 2)], fill=col, width=lw)
    if coords:
        f = font(st['coord_font'], int(big * st['coord_size']))
        files = 'abcdefgh' if orientation == 'white' else 'hgfedcba'
        ranks = '87654321' if orientation == 'white' else '12345678'
        off = fw + gap + (4 * SS if st['frame'] == 'double' else 0) + big * 0.12
        sides = ['bottom', 'left'] + (['top', 'right'] if st['coords'] == 'all' else [])
        for i in range(8):
            for side in sides:
                t = files[i] if side in ('top', 'bottom') else ranks[i]
                bb = f.getbbox(t)
                if side == 'bottom': tx, ty = x0 + i * big + big / 2 - (bb[2] + bb[0]) / 2, y0 + 8 * big + off - bb[1] * 0.3
                elif side == 'top': tx, ty = x0 + i * big + big / 2 - (bb[2] + bb[0]) / 2, y0 - off - bb[3]
                elif side == 'left': tx, ty = x0 - off - bb[2], y0 + i * big + big / 2 - (bb[3] + bb[1]) / 2
                else: tx, ty = x0 + 8 * big + off - bb[0], y0 + i * big + big / 2 - (bb[3] + bb[1]) / 2
                d.text((tx, ty), t, font=f, fill=(0, 0, 0))
    out = img.resize((img.width // SS, img.height // SS), Image.LANCZOS)
    return out, (margin / SS, margin / SS, 8 * p, 8 * p)

TINTS = {'dim': (206, 198, 186), 'sepia': (241, 226, 196), 'paper': (249, 243, 230)}

def degrade_book(img, kind, rnd):
    """→ (image, rotation, scale x, scale y): what happens to a diagram on its way into a snip"""
    rot, sx, sy = 0.0, 1.0, 1.0
    if kind == 'clean':
        return img, rot, sx, sy
    if kind in ('dim', 'sepia', 'paper'):
        # the reader's page colour multiplied in (the night theme dims pictures), seen at screen resolution and
        # sometimes stretched a little by whatever cropped or scaled the picture
        t = TINTS[kind]
        img = Image.merge('RGB', [ch.point(lambda v, k=k: v * k // 255) for ch, k in zip(img.split(), t)])
        sx = rnd.uniform(0.7, 1.0)
        sy = sx * (rnd.uniform(0.86, 1.0) if rnd.random() < 0.35 else 1.0)
        img = img.resize((max(20, int(img.width * sx)), max(20, int(img.height * sy))), Image.BICUBIC)
        if rnd.random() < 0.5:
            buf = io.BytesIO(); img.save(buf, 'JPEG', quality=rnd.randint(70, 92)); buf.seek(0)
            img = Image.open(buf).convert('RGB')
        return img, rot, sx, sy
    if kind == 'screen':
        sx = sy = rnd.uniform(0.55, 0.85)
        img = img.resize((max(20, int(img.width * sx)), max(20, int(img.height * sy))), Image.BICUBIC)
        return img, rot, sx, sy
    if kind == 'upscaled':
        # a small picture in a PDF rendered again at a high resolution (the snip of a PDF)
        s = rnd.uniform(0.45, 0.65)
        small = img.resize((max(20, int(img.width * s)), max(20, int(img.height * s))), Image.BICUBIC)
        img = small.resize(img.size, Image.BICUBIC)
        return img, rot, sx, sy
    im, rot, scale = degrade(img, kind, rnd)
    return im, rot, scale, scale

BOOK_DEGS = ['clean', 'clean', 'dim', 'dim', 'sepia', 'screen', 'upscaled', 'scan', 'jpeg', 'rot', 'bw']

def book(folder, n, seed=20261004, sets=None, degs=None, screen=0.0, first=0):
    """n diagrams in the classic book style and <folder>/truth.json (screen: share of screen boards in colour;
    first: number of the first file, so that several processes can fill one folder)"""
    os.makedirs(folder, exist_ok=True)
    rnd = random.Random(seed)
    sets = sets or (SETS + ['proc-%d' % k for k in range(900, 912)])
    degs = degs or BOOK_DEGS
    positions = book_positions(n, rnd)
    truth = []
    for i, (fen, kind, last) in enumerate(positions):
        s = sets[i % len(sets)] if rnd.random() < 0.5 else rnd.choice(sets)
        deg = rnd.choice(degs)
        orientation = 'black' if rnd.random() < 0.15 else 'white'
        st = book_style(rnd, screen)
        if st['dark'] in SCREEN_COLOURS and deg == 'bw': deg = 'clean'
        if orientation == 'black' and st['coords'] == 'none': st['coords'] = 'lb'
        p = {'screen': rnd.randint(34, 52), 'dim': rnd.randint(36, 60)}.get(deg, rnd.randint(30, 64))
        grid = grid_from_fen(fen, orientation)
        arrows = book_arrows(fen, last, rnd, orientation) if rnd.random() < 0.55 else []
        img, rect = draw_book(grid, s, p, st, rnd, orientation, arrows)
        stm = 'b' if ' b ' in fen else 'w'
        cap = rnd.choice(CAPTIONS_W if stm == 'w' else CAPTIONS_B).format(n=rnd.randint(1, 240)) if rnd.random() < 0.4 else None
        snip, brect = compose_snip(img, rect, rnd, caption=cap, text_above=rnd.random() < 0.25, text_below=rnd.random() < 0.2)
        snip, rot, sx, sy = degrade_book(snip, deg, rnd)
        name = f'b{first + i:05d}.png' if first else f'b{i:04d}.png'
        snip.save(os.path.join(folder, name))
        truth.append({'file': name, 'set': s, 'style': st['dark'], 'deg': deg, 'kind': kind, 'orientation': orientation, 'stm': stm,
                      'fen': fen, 'grid': grid, 'coords': st['coords'], 'arrows': len(arrows), 'p': p, 'rot': round(rot, 2),
                      'board': [round(brect[0] * sx, 2), round(brect[1] * sy, 2), round(brect[2] * sx, 2), round(brect[3] * sy, 2)]})
        if (i + 1) % 100 == 0: print(i + 1, 'diagrams', flush=True)
    json.dump(truth, open(os.path.join(folder, 'truth.json' if not first else f'truth-{first}.json'), 'w', encoding='utf8'), indent=0)
    print('book', len(truth), 'diagrams in', folder)

FIXTURE_FEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'

def fixtures(folder, set_name="proc-905"):
    """build/testlib/fixtures: one diagram after 1.e4 in the style of a printed chess book (a made-up diagram font,
    hatched dark squares, a thin frame, coordinates on four sides, arrows for 1.e4 and for the knight's answer), cut out
    of a page three times as the reader shows it: the page itself in grey (the picture of the test PDF of
    chess-snip-check), a night page that dims the picture and squeezes it a little in height, and the same night page
    cut out of the screen with the dark reader around it. fixtures.json gets the position and the board rectangle."""
    os.makedirs(folder, exist_ok=True)
    grid = grid_from_fen(FIXTURE_FEN, 'white')
    st = {'dark': 'hatch45', 'spacing': 7.5, 'line': 1.0, 'grey': (178, 178, 178), 'frame': 'thin', 'frame_gap': 0, 'coords': 'all',
          'coord_font': 'arial.ttf', 'coord_size': 0.3, 'arrow_w': 0.045, 'head_len': 0.24, 'head_w': 0.22, 'head_fill': True, 'arrow_grey': 0}
    # e2-e4: the pawn stands at the head, so it stops short; g8-f6: the head on the square
    arrows = [(4, 6, 4, 4, 0.42), (6, 0, 5, 2, 0.0)]
    p = 36
    img, rect = draw_book(grid, set_name, p, st, random.Random(1904), 'white', arrows)
    # the page around the diagram: the text column is wider than the diagram
    W, H = int(img.width * 1.95), int(img.height * 1.1)
    page = Image.new('RGB', (W, H), (255, 255, 255))
    ox, oy = (W - img.width) // 2, (H - img.height) // 2
    page.paste(img, (ox, oy))
    board = [round(ox + rect[0], 1), round(oy + rect[1], 1), round(ox + rect[0] + rect[2], 1), round(oy + rect[1] + rect[3], 1)]
    page.convert('L').save(os.path.join(folder, 'hatched-e4-arrows-page.png'), optimize=True)
    # the night page: the page colour multiplied in, the picture a little lower than wide (the reader scaled it)
    t = TINTS['dim']
    night = Image.merge('RGB', [ch.point(lambda v, k=k: v * k // 255) for ch, k in zip(page.split(), t)])
    night = night.resize((int(W * 1.02), int(H * 0.95)), Image.BICUBIC)
    night.save(os.path.join(folder, 'hatched-e4-arrows.png'), optimize=True)
    # the screen: the same picture with the dark reader around it
    pad = 18
    screen = Image.new('RGB', (night.width + 2 * pad, night.height + 2 * pad), (34, 33, 36))
    screen.paste(night, (pad, pad))
    screen.save(os.path.join(folder, 'hatched-e4-arrows-screen.png'), optimize=True)
    common = {'fen': FIXTURE_FEN, 'orientation': 'white', 'style': 'hatch45'}
    meta = {
        'hatched-e4-arrows.png': {**common, 'note': 'drawn by training/gen_chess.py fixtures: a diagram in the style of a printed chess book as a night page shows it (dimmed), cut out and squeezed a little in height'},
        'hatched-e4-arrows-screen.png': {**common, 'note': 'drawn by training/gen_chess.py fixtures: the same diagram cut out of the screen with the dark page around it'},
        'hatched-e4-arrows-page.png': {**common, 'board': board, 'note': 'drawn by training/gen_chess.py fixtures: the same diagram on a white page in grey (the picture of the test PDF in chess-snip-check); board: the board inside the picture (pixels)'},
    }
    with open(os.path.join(folder, 'fixtures.json'), 'w', encoding='utf8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print('fixtures in', folder, 'board', board, 'set', set_name)

def openings_module(out):
    """diagram/openings.js: the placements along the opening lines (a known opening fits a diagram
    better than a position with a few pieces somewhere else)"""
    import chess
    seen = []
    for fen, _ in opening_positions():
        pl = fen.split()[0]
        if pl not in seen: seen.append(pl)
    lines = [
        "// Positions along common opening lines (the start position included), from White's side: a diagram of a book that",
        '// shows one of them is read as that position when only a few unsure squares stand against it (decode.js).',
        '// Written by: python training/gen_chess.py openings diagram/openings.js',
        'export const OPENING_PLACEMENTS = [',
    ] + [f"  '{pl}'," for pl in seen] + ['];']
    with open(out, 'w', encoding='utf8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(len(seen), 'placements in', out)

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'openings':
        openings_module(sys.argv[2]); sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == 'fixtures':
        fixtures(os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, 'fixtures'), *(sys.argv[3:4])); sys.exit(0)
    if len(sys.argv) < 3 or sys.argv[1] not in ('bench', 'lib', 'book'):
        sys.exit('usage: gen_chess.py bench <folder> [per] | lib <folder> [epub folder] | book <folder> [n] [seed] [sets]')
    if sys.argv[1] == 'bench':
        bench(os.path.abspath(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    elif sys.argv[1] == 'book':
        # book <folder> [n] [seed] [sets a,b,…|proc:0-899] [screen share] [first file number]
        sets = None
        if len(sys.argv) > 5 and sys.argv[5] != '-':
            sets = []
            for part in sys.argv[5].split(','):
                if part.startswith('proc:'):
                    a, b = part[5:].split('-'); sets += ['proc-%d' % k for k in range(int(a), int(b) + 1)]
                elif part == 'ext:all':
                    # every set in the folder of CHESS_EXT_SETS (piece_images)
                    ext = os.environ.get('CHESS_EXT_SETS', '')
                    sets += ['ext-' + d for d in sorted(os.listdir(ext)) if os.path.exists(os.path.join(ext, d, 'wK.png'))]
                else: sets.append(part)
        book(os.path.abspath(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 300, int(sys.argv[4]) if len(sys.argv) > 4 else 20261004,
             sets, None, float(sys.argv[6]) if len(sys.argv) > 6 else 0.0, int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    else:
        pdf_lib(os.path.abspath(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
