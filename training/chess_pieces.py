# Made-up chess piece sets in the style of printed diagram fonts, for the tests and the training of the diagram
# recognition (Cozy Shelves 0.7). Nothing here is copied from a font: every set is drawn from a handful of style
# parameters (stroke width, proportions, base, bishop slit, crown points, knight profile, how black pieces show their
# inner lines …), so that the recognition can be measured on piece designs it has never seen.
#
#   piece_set(seed) -> style dict;  render_set(style, size) -> {'K': RGBA image, …, 'p': …}
#
# White pieces have an opaque white body with a black outline and black inner lines; black pieces are solid with
# white inner lines (or none), as in the classic diagram fonts.
import math, random
from PIL import Image, ImageDraw

YB = 0.93  # base line of every piece (share of the square from the top)


def _bez(p0, p1, p2, p3, n=14):
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
                    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]))
    return out


class P:
    """a tiny path builder: moves, lines and cubic curves, flattened to points"""
    def __init__(self, x, y):
        self.pts = [(x, y)]

    def L(self, x, y):
        self.pts.append((x, y))
        return self

    def C(self, x1, y1, x2, y2, x, y, n=14):
        self.pts += _bez(self.pts[-1], (x1, y1), (x2, y2), (x, y), n)
        return self


def ellipse(cx, cy, rx, ry, n=40, a0=0.0, a1=2 * math.pi):
    return [(cx + rx * math.cos(a0 + (a1 - a0) * i / n), cy + ry * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def mirror(half, cx=0.5):
    """a symmetric outline from its left half (top to bottom): the right half is the mirror image, reversed"""
    return half + [(2 * cx - x, y) for (x, y) in reversed(half)]


def rect(x0, y0, x1, y1, r=0.0):
    if r <= 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = []
    for (cx, cy, a) in ((x1 - r, y0 + r, -math.pi / 2), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, math.pi / 2), (x0 + r, y0 + r, math.pi)):
        pts += ellipse(cx, cy, r, r, 6, a, a + math.pi / 2)
    return pts


# ------------------------------------------------------------------------------------------------ styles
def piece_set(seed):
    """the style of one made-up set"""
    r = random.Random(seed)
    u = r.uniform
    return {
        'seed': seed,
        'sw': u(0.022, 0.05),                    # outline width (share of the square)
        'size': u(0.8, 0.94),                    # height of the king
        'fat': u(0.88, 1.14),                    # width factor
        'base': r.choice(['flat', 'step', 'step', 'round']),
        'base_w': u(0.6, 0.74),
        'base_h': u(0.06, 0.11),
        'pawn_h': u(0.52, 0.64), 'pawn_r': u(0.085, 0.118), 'pawn_collar': r.random() < 0.7, 'pawn_body': r.choice(['bell', 'cone', 'round']),
        'pawn_base_w': u(0.46, 0.6),
        'rook_h': u(0.66, 0.8), 'rook_m': r.choice([3, 3, 4, 5]), 'rook_notch': u(0.05, 0.1), 'rook_top': u(0.5, 0.62),
        'rook_body': u(0.38, 0.5), 'rook_taper': u(0.0, 0.06),
        'bishop_h': u(0.8, 0.9), 'bishop_w': u(0.26, 0.36), 'bishop_slit': r.choice(['diag', 'diag', 'cross', 'both', 'none']),
        'bishop_base': r.choice(['flat', 'ribbon', 'ribbon', 'base']), 'bishop_ball': u(0.03, 0.055), 'bishop_rings': r.choice([1, 1, 2]),
        'knight_seed': r.randint(0, 1 << 30), 'knight_h': u(0.8, 0.9), 'knight_base': r.random() < 0.65, 'knight_mane': r.random() < 0.7,
        'queen_h': u(0.82, 0.9), 'queen_n': r.choice([5, 5, 7, 9]), 'queen_balls': r.random() < 0.75, 'queen_w': u(0.6, 0.76),
        'queen_valley': u(0.12, 0.2),
        'king_h': 1.0, 'king_crown': r.choice(['lobes', 'lobes', 'cup']), 'king_cross': u(0.12, 0.18), 'king_cross_w': u(0.035, 0.055),
        'black': r.choice(['lines', 'lines', 'lines', 'solid', 'inline']),   # inner lines of black pieces
        'detail': u(0.6, 1.0),                   # width of inner lines relative to the outline
    }


# ------------------------------------------------------------------------------------------------ shapes
# every shape returns (parts, lines): parts are closed outlines drawn in order (white body with a black outline, or
# solid black), lines are open polylines for the inner details (black on white pieces, white on black ones)

def base_parts(st, w, h, kind=None):
    kind = kind or st['base']
    x0, x1 = 0.5 - w / 2, 0.5 + w / 2
    if kind == 'flat':
        return [rect(x0, YB - h, x1, YB, min(h, w) * 0.25)], []
    if kind == 'round':
        p = P(x0, YB).C(x0, YB - h * 0.7, 0.5 - w * 0.42, YB - h, 0.5 - w * 0.3, YB - h).L(0.5 + w * 0.3, YB - h).C(0.5 + w * 0.42, YB - h, x1, YB - h * 0.7, x1, YB)
        return [p.pts], []
    # two steps
    lo = rect(x0, YB - h * 0.55, x1, YB, h * 0.12)
    hi = rect(0.5 - w * 0.4, YB - h, 0.5 + w * 0.4, YB - h * 0.5, h * 0.1)
    return [lo, hi], []


def pawn(st):
    H, r = st['pawn_h'], st['pawn_r']
    bw, bh = st['pawn_base_w'], st['base_h'] * 0.9
    cy = YB - H + r
    parts, lines = [], []
    top = YB - bh
    yc = cy + r * 0.9
    wc = r * (2.3 if st['pawn_collar'] else 1.4)
    wt, wb = wc * 0.36, bw * 0.4
    if st['pawn_body'] == 'bell':
        half = P(0.5 - wt, yc).C(0.5 - wt, yc + (top - yc) * 0.45, 0.5 - wb * 0.95, top - (top - yc) * 0.25, 0.5 - wb, top).pts
    elif st['pawn_body'] == 'cone':
        half = [(0.5 - wt, yc), (0.5 - wb, top)]
    else:
        half = P(0.5 - wt, yc).C(0.5 - wb * 1.1, yc + (top - yc) * 0.3, 0.5 - wb * 1.05, top - (top - yc) * 0.2, 0.5 - wb * 0.95, top).pts
    parts.append(mirror(half))
    bp, _ = base_parts(st, bw, bh, 'flat' if st['base'] == 'step' else st['base'])
    parts += bp
    if st['pawn_collar']:
        parts.append(ellipse(0.5, yc, wc / 2, max(0.018, r * 0.32)))
    parts.append(ellipse(0.5, cy, r, r))
    return parts, lines


def rook(st):
    H = st['rook_h']
    bw, bh = st['base_w'], st['base_h']
    parts, lines = [], []
    tw = st['rook_top']
    th = 0.11 + st['rook_notch'] * 0.6
    ytop = YB - H
    yb_body = YB - bh
    yt_body = ytop + th
    w0, w1 = st['rook_body'] + st['rook_taper'], st['rook_body']
    parts.append([(0.5 - w0 / 2, yb_body + 0.01), (0.5 - w1 / 2, yt_body - 0.01), (0.5 + w1 / 2, yt_body - 0.01), (0.5 + w0 / 2, yb_body + 0.01)])
    bp, _ = base_parts(st, bw, bh)
    parts += bp
    # the top with its battlements
    m = st['rook_m']
    n = 2 * m - 1
    seg = tw / n
    nd = st['rook_notch']
    pts = [(0.5 - tw / 2, yt_body), (0.5 - tw / 2, ytop)]
    for k in range(n):
        x = 0.5 - tw / 2 + k * seg
        y = ytop if k % 2 == 0 else ytop + nd
        pts += [(x, y), (x + seg, y)]
    pts += [(0.5 + tw / 2, ytop), (0.5 + tw / 2, yt_body)]
    parts.append(pts)
    # a collar under the top and above the base
    lines.append([(0.5 - w1 / 2 * 0.98, yt_body + 0.035), (0.5 + w1 / 2 * 0.98, yt_body + 0.035)])
    lines.append([(0.5 - w0 / 2 * 0.98, yb_body - 0.04), (0.5 + w0 / 2 * 0.98, yb_body - 0.04)])
    return parts, lines


def bishop(st):
    H = st['bishop_h']
    parts, lines = [], []
    bw, bh = st['base_w'] * 0.92, st['base_h']
    kind = st['bishop_base']
    if kind == 'ribbon':
        # a bow under the bishop: two curls left and right of the foot
        y0 = YB - bh * 0.5
        left = P(0.5, YB - bh * 1.1).C(0.5 - bw * 0.2, YB - bh * 0.2, 0.5 - bw * 0.35, YB - bh * 1.2, 0.5 - bw * 0.5, YB - bh * 0.6) \
            .C(0.5 - bw * 0.56, YB - bh * 0.2, 0.5 - bw * 0.48, YB + 0.004, 0.5 - bw * 0.36, YB - bh * 0.15) \
            .C(0.5 - bw * 0.22, YB - bh * 0.45, 0.5 - bw * 0.1, YB + 0.002, 0.5, YB - bh * 0.25).pts
        right = [(1 - x, y) for (x, y) in reversed(left)]
        parts.append(left + right)
        lines.append([(0.5 - bw * 0.36, YB - bh * 0.15 - 0.01), (0.5 - bw * 0.46, YB - bh * 0.6)])
        lines.append([(0.5 + bw * 0.36, YB - bh * 0.15 - 0.01), (0.5 + bw * 0.46, YB - bh * 0.6)])
        top = YB - bh * 1.05
    elif kind == 'base':
        bp, _ = base_parts(st, bw, bh)
        parts += bp
        top = YB - bh
    else:
        parts.append(rect(0.5 - bw * 0.42, YB - bh * 0.8, 0.5 + bw * 0.42, YB, bh * 0.2))
        top = YB - bh * 0.8
    ball = st['bishop_ball']
    mw = st['bishop_w']
    ytip = YB - H + 2 * ball
    stem_h = 0.07 + (0.05 if kind == 'ribbon' else 0.03)
    ym0 = top - stem_h
    # stem and rings
    sw = mw * 0.5
    parts.append([(0.5 - sw / 2, top + 0.01), (0.5 - sw / 2 * 0.8, ym0), (0.5 + sw / 2 * 0.8, ym0), (0.5 + sw / 2, top + 0.01)])
    for k in range(st['bishop_rings']):
        parts.append(ellipse(0.5, ym0 + 0.012 - k * 0.04, mw * 0.5, 0.024))
    yb_m = ym0 - 0.02 - (st['bishop_rings'] - 1) * 0.04
    # the mitre: wide below, pointed above
    mh = yb_m - ytip
    half = P(0.5, ytip).C(0.5 - mw * 0.2, ytip + mh * 0.18, 0.5 - mw * 0.55, ytip + mh * 0.5, 0.5 - mw * 0.48, ytip + mh * 0.8) \
        .C(0.5 - mw * 0.44, ytip + mh * 0.95, 0.5 - mw * 0.3, yb_m, 0.5 - mw * 0.25, yb_m).pts
    parts.append(mirror(half))
    parts.append(ellipse(0.5, ytip - ball * 0.8, ball, ball))
    yc = ytip + mh * 0.55
    if st['bishop_slit'] in ('diag', 'both'):
        lines.append([(0.5 + mw * 0.3, ytip + mh * 0.28), (0.5 - mw * 0.02, ytip + mh * 0.58)])
    if st['bishop_slit'] in ('cross', 'both'):
        c = mw * 0.22
        lines.append([(0.5, yc - c), (0.5, yc + c)])
        lines.append([(0.5 - c, yc), (0.5 + c, yc)])
    return parts, lines


KNIGHT = [  # classic profile facing left (base line excluded), clockwise from the back of the base
    (0.78, 0.86), (0.79, 0.7), (0.77, 0.52), (0.72, 0.36), (0.64, 0.24), (0.58, 0.19), (0.57, 0.1), (0.52, 0.17),
    (0.46, 0.18), (0.41, 0.15), (0.38, 0.21), (0.3, 0.27), (0.22, 0.38), (0.15, 0.47), (0.13, 0.53), (0.16, 0.57),
    (0.22, 0.58), (0.27, 0.6), (0.33, 0.58), (0.41, 0.53), (0.47, 0.5), (0.45, 0.6), (0.38, 0.68), (0.31, 0.77),
    (0.27, 0.86),
]


def knight(st):
    r = random.Random(st['knight_seed'])
    H = st['knight_h']
    parts, lines = [], []
    bh = st['base_h'] if st['knight_base'] else 0.0
    if st['knight_base']:
        bp, _ = base_parts(st, st['base_w'] * 0.95, bh)
        parts += bp
    # jitter the profile a little and stretch it to the height of this set
    jit = [(x + r.uniform(-0.025, 0.025), y + r.uniform(-0.02, 0.02)) for (x, y) in KNIGHT]
    jit[0] = (jit[0][0], 0.86)
    jit[-1] = (jit[-1][0], 0.86)
    top = YB - bh + 0.01
    k = (H - bh) / (0.86 - 0.1)
    pts = [(0.5 + (x - 0.5) * (1 + r.uniform(-0.08, 0.08)), top - (0.86 - y) * k) for (x, y) in jit]
    # smooth the polygon (Chaikin) so that it reads as a drawn head, not a polygon
    for _ in range(2):
        sm = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if i == n - 1:
                sm.append(a)
                continue
            sm += [(0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]), (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])]
        pts = [pts[0]] + sm[1:] + [pts[-1]]
    parts.append(pts)
    def at(x, y):
        return (0.5 + (x - 0.5), top - (0.86 - y) * k)
    lines.append([at(0.36, 0.3), at(0.39, 0.29)])                 # eye
    lines.append([at(0.18, 0.5), at(0.2, 0.5)])                   # nostril
    if st['knight_mane']:
        lines.append([at(0.69, 0.33), at(0.73, 0.48), at(0.74, 0.66), at(0.73, 0.82)])
    return parts, lines


def body_and_band(st, H, w_band):
    """the body of queen and king from the base up to the band under the crown; returns (parts, lines, y of the band)"""
    bw, bh = st['base_w'], st['base_h']
    parts, lines = [], []
    bp, _ = base_parts(st, bw, bh)
    top = YB - bh
    yband = YB - H * 0.48
    wb = bw * 0.33
    half = P(0.5 - w_band * 0.42, yband).C(0.5 - w_band * 0.4, yband + (top - yband) * 0.5, 0.5 - wb * 1.05, top - (top - yband) * 0.15, 0.5 - wb * 1.15, top + 0.01).pts
    parts.append(mirror(half))
    parts += bp
    parts.append(ellipse(0.5, yband, w_band * 0.5, 0.028))
    lines.append([(0.5 - wb * 1.0, top - 0.035), (0.5 + wb * 1.0, top - 0.035)])
    return parts, lines, yband


def queen(st):
    H = st['queen_h']
    W = st['queen_w']
    parts, lines, yband = body_and_band(st, H, W * 0.62)
    n = st['queen_n']
    balls = st['queen_balls']
    rb = 0.032 if balls else 0.0
    ytip = YB - H + 2 * rb
    valley = st['queen_valley']
    pts = [(0.5 - W * 0.31, yband)]
    for i in range(n):
        t = i / (n - 1)
        x = 0.5 - W / 2 + W * t
        y = ytip + 0.07 * (2 * abs(t - 0.5)) ** 1.5      # outer points lower
        pts.append((x, y))
        if i < n - 1:
            xv = x + W / (n - 1) / 2
            pts.append((xv, y + valley * (1 - 0.3 * abs(t - 0.5))))
    pts.append((0.5 + W * 0.31, yband))
    parts.insert(len(parts) - 1, pts)
    if balls:
        for i in range(n):
            t = i / (n - 1)
            x = 0.5 - W / 2 + W * t
            y = ytip + 0.07 * (2 * abs(t - 0.5)) ** 1.5
            parts.append(ellipse(x, y - rb * 0.7, rb, rb))
    lines.append([(0.5 - W * 0.3, yband - 0.05), (0.5 + W * 0.3, yband - 0.05)])
    return parts, lines


def king(st):
    H = st['size']
    parts, lines, yband = body_and_band(st, H, 0.42)
    cross = st['king_cross']
    cw = st['king_cross_w']
    ytop = YB - H
    ycr = ytop + cross          # bottom of the cross
    if st['king_crown'] == 'lobes':
        rr = 0.13
        cyl = yband - rr * 0.95
        left = ellipse(0.5 - rr * 0.95, cyl, rr, rr * 0.95, 30)
        right = ellipse(0.5 + rr * 0.95, cyl, rr, rr * 0.95, 30)
        parts.insert(len(parts) - 1, left)
        parts.insert(len(parts) - 1, right)
        mid = [(0.5 - 0.055, yband), (0.5 - 0.06, ycr + 0.06), (0.5, ycr), (0.5 + 0.06, ycr + 0.06), (0.5 + 0.055, yband)]
        parts.insert(len(parts) - 1, mid)
    else:
        cup = [(0.5 - 0.2, yband), (0.5 - 0.3, ycr + 0.1), (0.5 - 0.12, ycr + 0.05), (0.5, ycr + 0.02), (0.5 + 0.12, ycr + 0.05), (0.5 + 0.3, ycr + 0.1), (0.5 + 0.2, yband)]
        parts.insert(len(parts) - 1, cup)
        lines.append([(0.5 - 0.2, ycr + 0.14), (0.5 + 0.2, ycr + 0.14)])
    # the cross
    parts.append(rect(0.5 - cw / 2, ytop, 0.5 + cw / 2, ycr + 0.02))
    parts.append(rect(0.5 - cross * 0.42, ytop + cross * 0.3, 0.5 + cross * 0.42, ytop + cross * 0.3 + cw))
    lines.append([(0.5 - 0.15, yband - 0.05), (0.5 + 0.15, yband - 0.05)])
    return parts, lines


SHAPES = {'p': pawn, 'r': rook, 'b': bishop, 'n': knight, 'q': queen, 'k': king}


def render_piece(st, ch, size, ss=4):
    """one piece of the set as an RGBA image size x size"""
    big = size * ss
    parts, lines = SHAPES[ch.lower()](st)
    white = ch.isupper()
    fat, sw = st['fat'], st['sw']
    # proportions of the set: width factor and the height of the king
    sc = st['size'] / 1.0

    def tx(p):
        x, y = p
        return ((0.5 + (x - 0.5) * fat * (0.92 + 0.08 * sc)) * big, (YB - (YB - y) * sc) * big)

    img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(1, int(round(sw * big)))
    ink, paper = (0, 0, 0, 255), (255, 255, 255, 255)
    for part in parts:
        pts = [tx(p) for p in part]
        if white:
            d.polygon(pts, fill=paper)
            d.line(pts + [pts[0]], fill=ink, width=w, joint='curve')
        else:
            d.polygon(pts, fill=ink)
            d.line(pts + [pts[0]], fill=ink, width=w, joint='curve')
    lw = max(1, int(round(sw * st['detail'] * big)))
    if white:
        for ln in lines:
            d.line([tx(p) for p in ln], fill=ink, width=lw, joint='curve')
    elif st['black'] != 'solid':
        for ln in lines:
            d.line([tx(p) for p in ln], fill=paper, width=max(1, int(lw * 0.8)), joint='curve')
        if st['black'] == 'inline':
            # a white line just inside the outline of the main part
            main = parts[0] if ch.lower() != 'k' else parts[-3]
            pts = [tx(p) for p in main]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            inner = [(cx + (x - cx) * 0.82, cy + (y - cy) * 0.86) for (x, y) in pts]
            d.line(inner + [inner[0]], fill=paper, width=max(1, int(lw * 0.6)), joint='curve')
    return img.resize((size, size), Image.LANCZOS)


def render_set(st, size):
    return {ch: render_piece(st, ch, size) for ch in 'KQRBNPkqrbnp'}


if __name__ == '__main__':
    # a contact sheet of a few sets: python chess_pieces.py out.png [first seed] [count]
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'pieces.png'
    s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    size = 64
    sheet = Image.new('RGB', (12 * size, n * size), (255, 255, 255))
    for row in range(n):
        st = piece_set(s0 + row)
        for k, ch in enumerate('KQRBNPkqrbnp'):
            im = render_piece(st, ch, size)
            bg = Image.new('RGBA', (size, size), (255, 255, 255, 255) if (k + row) % 2 == 0 else (200, 200, 200, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert('RGB'), (k * size, row * size))
    sheet.save(out)
