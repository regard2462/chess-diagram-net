# Checked labels for the squares of real books (training/scan-books.js): a book prints every piece with the
# same glyph, so the squares with something on them are grouped by their look (k-means on the cleaned square) and
# every group is shown on a sheet with the current network's reading (a group the network reads in two ways is split
# again, twice). A person checks the sheets and writes the right piece for every group whose reading is wrong;
# label.py apply then gives the squares of a group its label.
# A square the network reads differently from its group keeps the group's label only in a group the network reads
# alike (nine in ten) and when it lies as close to the centre as half the group (the same glyph, misread); else it is
# dropped (another glyph that fell into the group).
#   python training/label.py sheets <prefix>[,<prefix>…] <out folder> [--per 25] [--seed 1] [--logits l]
#       (--logits l2: the readings of a newer network, relabel.py)
#       → <out>/clusters.json, <out>/overview.png (per reading the mean picture of every clear group) and
#       <out>/sheet-<n>.png (the other groups: 12 random squares and up to 6 read differently, those framed)
#   python training/label.py apply <prefix>[,<prefix>…] <out folder> [<decisions.json>]
#       decisions: { "<group>": "<class>" (every square of the group is that piece, checked by eye) | "x" (drop the
#       group) | "bad" (drop every diagram with a square in the group: a board found in the wrong place) }; groups
#       without a decision keep the network's reading as described above. Writes <prefix>.y (checked classes, 255 = dropped) and
#       <prefix>.lab.json ({ labels: per diagram the 64 classes, '?' where a square was dropped, or null }).
#   python training/label.py check <prefix>[,…] <out folder> <decisions.json>
#       → <out>/check-<n>.png: every group with a class or 'bad' decision, with its squares (to check the decisions)
# Needs numpy and Pillow.
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
NAMES = {'.': 'empty', 'P': 'white pawn', 'N': 'white knight', 'B': 'white bishop', 'R': 'white rook', 'Q': 'white queen', 'K': 'white king',
         'p': 'black pawn', 'n': 'black knight', 'b': 'black bishop', 'r': 'black rook', 'q': 'black queen', 'k': 'black king'}


def load(prefixes, logits='l'):
    out = []
    for p in prefixes:
        info = json.load(open(p + '.json', encoding='utf8'))
        n, S = info['n'], info['N']
        X = np.fromfile(p + '.x', dtype=np.uint8).reshape(n, 2, S, S)
        L = np.fromfile(p + '.' + logits, dtype=np.float32).reshape(n, 13)
        out.append((p, info, X, L))
    return out


def kmeans(F, k, seed, iters=25):
    rng = np.random.default_rng(seed)
    C = [F[rng.integers(len(F))]]
    d2 = ((F - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / max(d2.sum(), 1e-9)
        C.append(F[rng.choice(len(F), p=p)])
        d2 = np.minimum(d2, ((F - C[-1]) ** 2).sum(1))
    C = np.array(C)
    FF = (F ** 2).sum(1)[:, None]
    for _ in range(iters):
        D = FF - 2 * F @ C.T + (C ** 2).sum(1)[None]
        a = D.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = F[m].mean(0)
    D = FF - 2 * F @ C.T + (C ** 2).sum(1)[None]
    a = D.argmin(1)
    return a, np.sqrt(np.maximum(D[np.arange(len(F)), a], 0))


def piece_like(info, L):
    p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    occ = np.array([m[3] for m in info['meta']]) if info['meta'] else np.zeros(0)
    return (p.argmax(1) != 0) | (occ == 1) | (p[:, 0] < 0.9), p


def features(X):
    ink = 1 - X[:, 0].astype(np.float32) / 255
    return ink.reshape(len(X), -1)


def sheets(prefixes, out, per=25, seed=1, logits='l'):
    os.makedirs(out, exist_ok=True)
    data = load(prefixes, logits)
    feats, refs, probs = [], [], []
    for b, (p, info, X, L) in enumerate(data):
        if not info['n']: continue
        like, pr = piece_like(info, L)
        idx = np.where(like)[0]
        light = np.array([info['meta'][i][2] for i in idx], dtype=np.float32)[:, None]
        feats.append(np.hstack([features(X[idx]), light * 2.0]))
        refs += [(b, int(i)) for i in idx]
        probs.append(pr[idx])
    if not feats:
        print('no squares'); return
    F = np.vstack(feats); P = np.vstack(probs)
    k = int(min(140, max(4, len(F) // per)))
    a, dist = kmeans(F, k, seed)
    # a group the network reads in two ways is often two glyphs that look alike in a poor print (a king and a
    # bishop): such groups are split again, twice, so that each part shows one glyph
    parts = [(np.where(a == j)[0], dist) for j in range(k)]
    for depth in range(2):
        nxt = []
        for m, dd in parts:
            if len(m) < 8: nxt.append((m, dd)); continue
            read = P[m].argmax(1)
            if np.bincount(read, minlength=13).max() >= 0.95 * len(m): nxt.append((m, dd)); continue
            sa, sd = kmeans(F[m], int(min(8, max(2, len(m) // 6))), seed + 1 + depth)
            d2 = dd.copy(); d2[m] = sd
            for j in range(sa.max() + 1):
                if (sa == j).any(): nxt.append((m[sa == j], d2))
        parts = nxt
    groups = []
    for m, dd in parts:
        if not len(m): continue
        read = P[m].argmax(1)
        votes = np.bincount(read, minlength=13)
        lab = int(votes.argmax())
        agree = read == lab
        scale = float(np.median(dd[m][agree])) if agree.any() else 0.0
        groups.append({'id': len(groups), 'read': CLASSES[lab], 'n': int(len(m)), 'purity': round(float(agree.mean()), 3),
                       'conf': round(float(P[m, lab].mean()), 3), 'scale': round(scale, 4),
                       'members': [[refs[i][0], refs[i][1], int(read[t]), round(float(dd[i]), 4)] for t, i in enumerate(m)]})
    order = sorted(groups, key=lambda g: (CLASSES.index(g['read']), -g['n']))
    json.dump({'prefixes': prefixes, 'logits': logits, 'groups': groups}, open(os.path.join(out, 'clusters.json'), 'w'))
    rng = np.random.default_rng(seed)
    cell, rows_per = 40, 30
    try: font = ImageFont.truetype('arial.ttf', 13); small = ImageFont.truetype('arial.ttf', 10)
    except Exception: font = small = ImageFont.load_default()
    # the overview: per reading one line with the mean picture of every group (a group of one glyph gives a sharp
    # mean, a mixed one a ghost); groups the network is unsure about or split on get a line of their own below
    clear = [g for g in order if g['purity'] >= 0.97 and g['conf'] >= 0.8 and g['read'] != '.']
    trouble = [g for g in order if g not in clear]
    per_line = 20
    lines = []
    for c in CLASSES[1:]:
        gs = [g for g in clear if g['read'] == c]
        for s in range(0, len(gs), per_line): lines.append((c, gs[s:s + per_line]))
    if lines:
        img = Image.new('RGB', (150 + per_line * (cell + 6), len(lines) * (cell + 18) + 6), 'white')
        dr = ImageDraw.Draw(img)
        for r, (c, gs) in enumerate(lines):
            y = 4 + r * (cell + 18)
            dr.text((4, y + 12), f"{c} {NAMES[c]}", fill=(0, 0, 0), font=font)
            for t, g in enumerate(gs):
                tiles = np.stack([data[b][2][i, 0].astype(np.float32) for b, i, _, _ in g['members']])
                mean = Image.fromarray(tiles.mean(0).clip(0, 255).astype(np.uint8)).resize((cell, cell), Image.BILINEAR)
                x = 150 + t * (cell + 6)
                img.paste(mean, (x, y))
                dr.text((x, y + cell + 1), f"{g['id']}·{g['n']}", fill=(60, 60, 200), font=small)
            dr.line([(0, y + cell + 15), (img.width, y + cell + 15)], fill=(210, 210, 210))
        img.save(os.path.join(out, 'overview.png'))
    nsheet = 0
    for s in range(0, len(trouble), rows_per):
        part = trouble[s:s + rows_per]
        img = Image.new('RGB', (230 + 18 * (cell + 2), len(part) * (cell + 4) + 4), 'white')
        dr = ImageDraw.Draw(img)
        for r, g in enumerate(part):
            y = 4 + r * (cell + 4)
            dr.text((4, y + 4), f"#{g['id']} {g['read']} {NAMES[g['read']]}", fill=(0, 0, 0), font=font)
            dr.text((4, y + 21), f"n={g['n']} same={g['purity']:.2f} p={g['conf']:.2f}", fill=(90, 90, 90), font=font)
            mem = g['members']
            lab = CLASSES.index(g['read'])
            same = [t for t, m in enumerate(mem) if m[2] == lab]
            other = sorted([t for t, m in enumerate(mem) if m[2] != lab], key=lambda t: mem[t][3])
            # a small group is shown whole
            pick = list(range(len(mem))) if len(mem) <= 18 else [t for t in rng.permutation(same)[:12]] + other[:6]
            for c, t in enumerate(pick):
                b, i = mem[t][0], mem[t][1]
                tile = Image.fromarray(data[b][2][i, 0]).resize((cell, cell), Image.NEAREST).convert('RGB')
                x = 230 + c * (cell + 2)
                img.paste(tile, (x, y))
                if mem[t][2] != lab:
                    # read differently: red when it would be dropped, blue when it keeps the group's label
                    keep = g['purity'] >= 0.9 and mem[t][3] <= g['scale']
                    dr.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(40, 90, 230) if keep else (230, 30, 30), width=2)
                    dr.text((x + 2, y + 1), CLASSES[mem[t][2]], fill=(40, 90, 230) if keep else (230, 30, 30), font=font)
            dr.line([(0, y + cell + 2), (img.width, y + cell + 2)], fill=(200, 200, 200))
        nsheet += 1
        img.save(os.path.join(out, f'sheet-{nsheet}.png'))
    print(f'{len(F)} squares in {len(groups)} groups, {nsheet} sheets in {out}')


def check(prefixes, out, decisions_file):
    """sheets of the decided groups (a class or 'bad'), each row the decision and 16 of the group's squares: a
    decision written for the wrong group shows at once"""
    cl = json.load(open(os.path.join(out, 'clusters.json'), encoding='utf8'))
    data = load(prefixes, cl.get('logits', 'l'))
    dec = json.load(open(decisions_file, encoding='utf8'))
    rows = [(g, dec[str(g['id'])]) for g in cl['groups'] if dec.get(str(g['id']), 'x') != 'x']
    rows.sort(key=lambda r: (r[1] == 'bad', CLASSES.index(r[1]) if r[1] in CLASSES else 99, -r[0]['n']))
    rng = np.random.default_rng(1)
    cell, per = 40, 30
    try: font = ImageFont.truetype('arial.ttf', 13)
    except Exception: font = ImageFont.load_default()
    n = 0
    for s in range(0, len(rows), per):
        part = rows[s:s + per]
        img = Image.new('RGB', (250 + 16 * (cell + 2), len(part) * (cell + 4) + 4), 'white')
        dr = ImageDraw.Draw(img)
        for r, (g, d) in enumerate(part):
            y = 4 + r * (cell + 4)
            dr.text((4, y + 4), f"#{g['id']} -> {d} {NAMES.get(d, 'BAD: drop diagrams')}", fill=(0, 0, 160), font=font)
            dr.text((4, y + 21), f"n={g['n']} network read {g['read']}", fill=(90, 90, 90), font=font)
            for c, t in enumerate(rng.permutation(len(g['members']))[:16]):
                b, i = g['members'][t][0], g['members'][t][1]
                img.paste(Image.fromarray(data[b][2][i, 0]).resize((cell, cell), Image.NEAREST).convert('RGB'), (250 + c * (cell + 2), y))
            dr.line([(0, y + cell + 2), (img.width, y + cell + 2)], fill=(200, 200, 200))
        n += 1
        img.save(os.path.join(out, f'check-{n}.png'))
    print(len(rows), 'decided groups on', n, 'check sheets')


def apply(prefixes, out, decisions_file=None):
    cl = json.load(open(os.path.join(out, 'clusters.json'), encoding='utf8'))
    data = load(prefixes, cl.get('logits', 'l'))
    dec = json.load(open(decisions_file, encoding='utf8')) if decisions_file and os.path.exists(decisions_file) else {}
    labels, bad = [], [set() for _ in data]
    for b, (p, info, X, L) in enumerate(data):
        like, pr = piece_like(info, L) if info['n'] else (np.zeros(0, bool), None)
        labels.append(np.where(like, 255, 0).astype(np.uint8))   # not piece-like: empty
    for g in cl['groups']:
        checked = str(g['id']) in dec
        d = dec.get(str(g['id']), g['read'])
        lab = CLASSES.index(g['read'])
        for b, i, read, dist in g['members']:
            if d == 'bad': bad[b].add(data[b][1]['meta'][i][0]); continue
            if d == 'x': labels[b][i] = 255; continue
            # a group checked by eye takes its piece whole; else the network's own reading of the group, or a square as
            # close to the centre as the group's own
            labels[b][i] = CLASSES.index(d) if (checked or read == lab or (g['purity'] >= 0.9 and dist <= g['scale'])) else 255
    # the rules check the groups: a group whose squares often give a pawn on the first or last rank or a second king of
    # one colour holds another piece (or several) and is dropped whole
    group_of = {}
    for g in cl['groups']:
        for b, i, read, dist in g['members']: group_of[(b, i)] = g
    viol = {}
    for b, (p, info, X, L) in enumerate(data):
        cells = {}
        for i, m in enumerate(info['meta']):
            if m[0] in bad[b] or labels[b][i] == 255: continue
            cells.setdefault(m[0], []).append((m[1], CLASSES[labels[b][i]], i))
        for di, cs in cells.items():
            kings = {'K': [c for c in cs if c[1] == 'K'], 'k': [c for c in cs if c[1] == 'k']}
            for sq, ch, i in cs:
                if (ch in 'pP' and (sq < 8 or sq >= 56)) or (ch in 'Kk' and len(kings[ch]) > 1):
                    g = group_of.get((b, i))
                    if g is not None: viol[g['id']] = viol.get(g['id'], 0) + 1
    for g in cl['groups']:
        if viol.get(g['id'], 0) >= max(2, 0.1 * g['n']):
            print('  group', g['id'], 'breaks the rules', viol[g['id']], 'times in', g['n'], 'squares: dropped')
            for b, i, read, dist in g['members']: labels[b][i] = 255
    dropped = 0
    for b, (p, info, X, L) in enumerate(data):
        y = labels[b]
        for i, m in enumerate(info['meta']):
            if m[0] in bad[b]: y[i] = 255
        y.tofile(p + '.y')
        lab = []
        for di, dg in enumerate(info['diagrams']):
            if di in bad[b]: lab.append(None); continue
            cells = ['.'] * 64   # squares left out of the samples are clearly empty
            for i, m in enumerate(info['meta']):
                if m[0] == di: cells[m[1]] = '?' if y[i] == 255 else CLASSES[y[i]]
            lab.append(''.join(cells))
        json.dump({'labels': lab}, open(p + '.lab.json', 'w'))
        dropped += int((y == 255).sum())
    print('dropped', dropped, 'squares,', sum(len(s) for s in bad), 'diagrams marked bad')


if __name__ == '__main__':
    a = sys.argv
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    if a[1] == 'sheets': sheets(a[2].split(','), a[3], int(opt('--per', 25)), int(opt('--seed', 1)), opt('--logits', 'l'))
    elif a[1] == 'apply': apply(a[2].split(','), a[3], a[4] if len(a) > 4 else None)
    elif a[1] == 'check': check(a[2].split(','), a[3], a[4])
