# Training shards and test diagrams from the checked squares of real books (harvest.py, label.py apply).
#   python training/pack.py train <harvest folder> <book id>[,<book id>…] <out prefix>
#       → <out>.{x,s,y,json} in the format of extract.js (dropped squares left out; set 'real-<book>')
#   python training/pack.py test <harvest folder> <books folder> <book id>[,…] <out folder> [--max 60]
#       → the diagrams again as snips (the page rendered at the same zoom, cut out the same way) and
#       <out>/truth.json for training/bench.js ('?' marks a square without a checked label)
# Needs numpy (train) and PyMuPDF (test).
import json, os, re, sys
import numpy as np

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']


def chunks(hdir, bid):
    d = os.path.join(hdir, bid)
    ks = sorted(int(f[1:-5]) for f in os.listdir(d) if f.startswith('c') and f.endswith('.json') and not f.endswith('.lab.json'))
    return [os.path.join(d, f'c{k}') for k in ks]


def train(hdir, ids, out):
    X, S, Y, meta = [], [], [], []
    N = ns = None
    for bid in ids:
        for p in chunks(hdir, bid):
            info = json.load(open(p + '.json', encoding='utf8'))
            n = info['n']
            if not n or not os.path.exists(p + '.lab.json'): continue
            N, ns = info['N'], info['scalars']
            x = np.fromfile(p + '.x', dtype=np.uint8).reshape(n, 2, N, N)
            s = np.fromfile(p + '.s', dtype=np.float32).reshape(n, ns)
            y = np.fromfile(p + '.y', dtype=np.uint8)
            keep = y != 255
            X.append(x[keep]); S.append(s[keep]); Y.append(y[keep])
            k = os.path.basename(p)
            meta += [[len(meta) + t, f'b{bid}-{k}-d{m[0]}', f'real-{bid}', 'real', 'real', m[1]] for t, m in enumerate(np.array(info['meta'], dtype=object)[keep].tolist())]
    X = np.concatenate(X); S = np.concatenate(S); Y = np.concatenate(Y)
    X.tofile(out + '.x'); S.tofile(out + '.s'); Y.tofile(out + '.y')
    json.dump({'n': int(len(Y)), 'N': N, 'scalars': ns, 'classes': CLASSES, 'meta': meta}, open(out + '.json', 'w'))
    print(out, len(Y), 'squares, classes', np.bincount(Y, minlength=13).tolist())


def test(hdir, root, ids, out, maxn=60):
    import fitz
    os.makedirs(out, exist_ok=True)
    books = json.load(open(os.path.join(hdir, 'books.json'), encoding='utf8'))
    truth = []
    for bid in ids:
        doc = fitz.open(os.path.join(root, books[bid]['file']))
        got = 0
        for p in chunks(hdir, bid):
            info = json.load(open(p + '.json', encoding='utf8'))
            if not os.path.exists(p + '.lab.json'): continue
            labs = json.load(open(p + '.lab.json', encoding='utf8'))['labels']
            for di, (d, lab) in enumerate(zip(info['diagrams'], labs)):
                if lab is None or got >= maxn: continue
                pg = doc[d['page']]
                pix = pg.get_pixmap(matrix=fitz.Matrix(d['zoom'], d['zoom']), colorspace=fitz.csGRAY, alpha=False)
                x, y, w, h = d['crop']
                from PIL import Image
                img = Image.frombytes('L', (pix.width, pix.height), pix.samples).crop((x, y, x + w, y + h))
                name = f'{bid}-{os.path.basename(p)}-{di}.png'
                img.save(os.path.join(out, name))
                grid = [lab[r * 8:(r + 1) * 8] for r in range(8)]
                xs, ys = d['xs'], d['ys']
                place = '/'.join(re.sub(r'\.+', lambda m: str(len(m.group())), row.replace('?', '.')) for row in grid)
                truth.append({'file': name, 'set': f'real-{bid}', 'style': 'real', 'deg': 'real', 'kind': 'book', 'orientation': 'white', 'stm': 'w',
                              'fen': place + ' w - - 0 1', 'grid': grid, 'board': [xs[0], ys[0], xs[8] - xs[0], ys[8] - ys[0]], 'unknown': lab.count('?')})
                got += 1
        print(bid, got, 'diagrams')
    json.dump(truth, open(os.path.join(out, 'truth.json'), 'w'), indent=0)
    print(len(truth), 'test diagrams in', out)


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'train': train(a[2], a[3].split(','), a[4])
    elif a[1] == 'test': test(a[2], a[3], a[4].split(','), a[5], int(a[a.index('--max') + 1]) if '--max' in a else 60)
