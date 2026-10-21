# Trains the square classifier of the diagram recognition (diagram/net.js) on the samples written by
# training/extract.js and stores it as diagram/net-weights.js: int8 weights with one scale per
# output channel, batch normalisation folded into the convolutions. Needs Python with numpy and PyTorch (CPU is fine).
#   python training/train.py <prefix>[,<prefix>…] [--test <prefix>,…] [--epochs 14] [--size 32]
#                                   [--out diagram/net-weights.js] [--hold set,set] [--seed 1]
#                                   [--weight real-=4,proc-=1] [--ch 16,32,48,64] [--hidden 96]
# --hold leaves piece sets out of the training (they are measured separately: sets the network has never seen).
# --weight draws the samples of the sets whose name starts with a prefix that much more often (real book squares are
# fewer than generated ones). The shards may hold 48 or 32 pixel squares; all are brought to --size.
import argparse, base64, json, os, sys, time, zlib
import numpy as np
# torch 2.1 looks for the transformers package when it builds an optimizer and fails on a newer one: keep it out
sys.modules.setdefault('transformers', None)
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
CH = [16, 32, 48, 64]
HIDDEN = 96


class Net(nn.Module):
    def __init__(self, ns, size=48, ch=CH, hidden=HIDDEN, nclass=len(CLASSES)):
        super().__init__()
        layers, cin = [], 2
        for c in ch:
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        side = size >> len(ch)
        self.fc1 = nn.Linear(cin * side * side + ns, hidden)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, nclass)

    def forward(self, x, s):
        z = torch.cat([self.features(x).flatten(1), s], 1)
        return self.fc2(self.drop(F.relu(self.fc1(z))))


def pool_to(x, size):
    """uint8 squares (n, 2, N, N) → (n, 2, size, size), the area average of net.js and adaptive_avg_pool2d"""
    N = x.shape[-1]
    if N == size: return x
    out = np.empty(x.shape[:2] + (size, size), dtype=np.float32)
    f = N / size
    for y in range(size):
        y0, y1 = int(np.floor(y * f)), int(np.ceil((y + 1) * f))
        for xx in range(size):
            x0, x1 = int(np.floor(xx * f)), int(np.ceil((xx + 1) * f))
            out[:, :, y, xx] = x[:, :, y0:y1, x0:x1].reshape(x.shape[0], 2, -1).mean(-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def load(prefixes, size=None):
    """all shards into one array each, the squares at `size` pixels (the first shard's size without one)"""
    infos = [json.load(open(p + '.json')) for p in prefixes]
    total = sum(i['n'] for i in infos)
    N, ns = size or infos[0]['N'], infos[0]['scalars']
    X = np.empty((total, 2, N, N), dtype=np.uint8)
    S = np.empty((total, ns), dtype=np.float32)
    Y = np.empty(total, dtype=np.int64)
    metas, o = [], 0
    for p, info in zip(prefixes, infos):
        n = info['n']
        X[o:o + n] = pool_to(np.fromfile(p + '.x', dtype=np.uint8).reshape(n, 2, info['N'], info['N']), N)
        S[o:o + n] = np.fromfile(p + '.s', dtype=np.float32).reshape(n, ns)
        Y[o:o + n] = np.fromfile(p + '.y', dtype=np.uint8)
        metas += [[p] + m for m in info['meta']]
        o += n
    return X, S, Y, metas


def to_input(xb, size, dev='cpu'):
    x = torch.from_numpy(xb).to(dev).float()
    clean = x[:, :1] / 255.0
    resid = x[:, 1:] / 127.5 - 1.0
    t = torch.cat([clean, resid], 1)
    if t.shape[-1] != size:
        t = F.adaptive_avg_pool2d(t, size)
    return t


def augment(t, gen, strong=False):
    """small affine jitter (shift, scale, rotation), contrast and noise: what scans and resampling do"""
    b = t.shape[0]
    ang = (torch.rand(b, generator=gen) - 0.5) * 0.07
    sc = 1 + (torch.rand(b, generator=gen) - 0.5) * 0.16
    tx = (torch.rand(b, generator=gen) - 0.5) * 0.12
    ty = (torch.rand(b, generator=gen) - 0.5) * 0.12
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).to(t.device)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    # outside the square: paper (clean 1) and no difference (0)
    shifted = t.clone()
    shifted[:, 0] -= 1.0
    out = F.grid_sample(shifted, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    out[:, 0] += 1.0
    k = (0.75 + 0.4 * torch.rand(b, 1, 1, generator=gen)).to(t.device)
    out[:, 0] = 1 - (1 - out[:, 0]) * k
    out[:, 1] = out[:, 1] * k
    if strong:
        out = print_effects(out, gen)
    out = out + (torch.randn(out.shape, generator=gen) * (0.035 * torch.rand(b, 1, 1, 1, generator=gen))).to(t.device)
    out[:, 0].clamp_(0, 1)
    out[:, 1].clamp_(-1, 1)
    return out


def print_effects(out, gen):
    """what printing and scanning do to a glyph, each on a share of the squares: a blur (a soft scan), ink that
    spreads or thins (a bolder or lighter print: the dark parts of the square grow or shrink by up to a pixel), and a
    black and white scan (the cleaned square cut at a threshold)"""
    b, dev = out.shape[0], out.device
    pick = lambda share: (torch.rand(b, generator=gen) < share).to(dev).view(b, 1, 1, 1).float()
    # blur: a 3 × 3 binomial kernel mixed in with a random weight
    kern = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=dev) / 16
    blurred = F.conv2d(F.pad(out, (1, 1, 1, 1), mode='replicate'), kern.expand(2, 1, 3, 3), groups=2)
    wb = pick(0.3) * torch.rand(b, 1, 1, 1, generator=gen).to(dev)
    out = out * (1 - wb) + blurred * wb
    # ink spreading (the dark of the clean square grows: a minimum filter; the difference grows: a maximum) or thinning
    dmin = -F.max_pool2d(-F.pad(out[:, :1], (1, 1, 1, 1), value=1.0), 3, 1)
    dmax = F.max_pool2d(F.pad(out[:, :1], (1, 1, 1, 1), value=1.0), 3, 1)
    rmax = F.max_pool2d(F.pad(out[:, 1:], (1, 1, 1, 1), value=0.0), 3, 1)
    rmin = -F.max_pool2d(-F.pad(out[:, 1:], (1, 1, 1, 1), value=0.0), 3, 1)
    bold, thin = pick(0.15), pick(0.12)
    wa = 0.25 + 0.5 * torch.rand(b, 1, 1, 1, generator=gen).to(dev)
    out = torch.cat([
        out[:, :1] * (1 - wa * bold - wa * thin) + dmin * wa * bold + dmax * wa * thin,
        out[:, 1:] * (1 - wa * bold - wa * thin) + rmax * wa * bold + rmin * wa * thin], 1)
    # a black and white scan
    bw = pick(0.08)
    th = (0.35 + 0.3 * torch.rand(b, 1, 1, 1, generator=gen)).to(dev)
    out = torch.cat([out[:, :1] * (1 - bw) + (out[:, :1] > th).float() * bw, out[:, 1:]], 1)
    return out


def evaluate(model, X, S, Y, size, idx=None, bs=1024, dev='cpu'):
    model.eval()
    idx = np.arange(len(Y)) if idx is None else idx
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            lo = model(to_input(X[j], size, dev), torch.from_numpy(S[j]).to(dev))
            p = F.softmax(lo, 1).cpu()
            probs.append(p.numpy()); preds.append(p.argmax(1).numpy())
    preds = np.concatenate(preds) if preds else np.zeros(0, int)
    probs = np.concatenate(probs) if probs else np.zeros((0, len(CLASSES)))
    return preds, probs


def report(name, preds, Y, metas=None, groups=None):
    acc = (preds == Y).mean() if len(Y) else 0
    pieces = Y != 0
    line = f'{name}: {len(Y)} squares, accuracy {acc * 100:.2f} %, pieces {((preds == Y)[pieces]).mean() * 100:.2f} %'
    false_piece = ((preds != 0) & (Y == 0)).sum(); missed = ((preds == 0) & (Y != 0)).sum()
    line += f', false pieces {false_piece}, missed {missed}'
    print(line, flush=True)
    if groups is not None:
        for g in sorted(set(groups)):
            m = np.array([x == g for x in groups])
            if m.sum(): print(f'   {g:12s} {m.sum():7d}  {((preds == Y)[m]).mean() * 100:.2f} %')


def export(model, size, ns, path, info):
    """batch norm folded into the convolutions, int8 per output channel"""
    model.eval()
    model = model.cpu()
    layers = []
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    bns = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]
    def q(w):
        w = w.reshape(w.shape[0], -1)
        scale = np.maximum(np.abs(w).max(1), 1e-8) / 127.0
        qi = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
        return qi, scale
    for conv, bn in zip(convs, bns):
        g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        w = conv.weight.detach().numpy() * g[:, None, None, None]
        b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        qi, scale = q(w)
        layers.append({'kind': 'conv', 'shape': list(conv.weight.shape), 'scale': [float(f'{v:.6g}') for v in scale], 'bias': [float(f'{v:.6g}') for v in b], 'data': base64.b64encode(qi.tobytes()).decode()})
    for fc in (model.fc1, model.fc2):
        w = fc.weight.detach().numpy()
        qi, scale = q(w)
        layers.append({'kind': 'fc', 'shape': list(w.shape), 'scale': [float(f'{v:.6g}') for v in scale], 'bias': [float(f'{v:.6g}') for v in fc.bias.detach().numpy()], 'data': base64.b64encode(qi.tobytes()).decode()})
    model_json = {'version': info['version'], 'input': size, 'classes': CLASSES, 'scalars': ns, 'layers': layers}
    with open(path, 'w', encoding='utf8') as f:
        f.write('// The trained square classifier of the diagram recognition (net.js), written by training/train.py.\n')
        f.write('// ' + info['note'] + '\n')
        f.write('export default ' + json.dumps(model_json, separators=(',', ':')) + ';\n')
    print('wrote', path, os.path.getsize(path), 'bytes')


def main():
    # the console of Windows is not always UTF-8
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('train')
    ap.add_argument('--test', default='')
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hold', default='')
    ap.add_argument('--seed', type=int, default=1)
    # more threads than free cores make PyTorch on the CPU many times slower (measured: 4 threads 0.2 s per step,
    # 8 threads 3 s on a busy machine)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default='')
    ap.add_argument('--save', default='')
    ap.add_argument('--weight', default='')
    ap.add_argument('--ch', default=','.join(map(str, CH)))
    ap.add_argument('--hidden', type=int, default=HIDDEN)
    ap.add_argument('--init', default='')
    # samples drawn per epoch (with --weight); 0: as many as there are training samples
    ap.add_argument('--epoch-size', type=int, default=0)
    # print effects in the augmentation (blur, bolder or thinner ink, black and white scans)
    ap.add_argument('--strong', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, S, Y, meta = load(args.train.split(','), args.size)
    ns = S.shape[1]
    sets = np.array([m[3] for m in meta])
    hold = set(args.hold.split(',')) if args.hold else set()
    # validation: one diagram in ten (by its file name), never a held-out set
    dia = np.array([zlib.crc32((m[0] + m[2]).encode()) % 10 for m in meta])
    held = np.array([s in hold for s in sets])
    tr = np.where((dia != 0) & ~held)[0]
    va = np.where((dia == 0) & ~held)[0]
    ho = np.where(held)[0]
    print(f'{len(Y)} samples: train {len(tr)}, validation {len(va)}, held-out sets {len(ho)}; classes', np.bincount(Y, minlength=len(CLASSES)).tolist(), flush=True)
    model = Net(ns, args.size, [int(c) for c in args.ch.split(',')], args.hidden).to(dev)
    if args.init: model.load_state_dict(torch.load(args.init, map_location=dev))
    # how often each training sample is drawn: by the prefix of its set's name
    wt = np.ones(len(tr))
    for part in filter(None, args.weight.split(',')):
        pre, f = part.split('=')
        wt[np.array([sets[k].startswith(pre) for k in tr])] = float(f)
    wt /= wt.sum()
    print('device', dev, 'draws per set kind:', {pre: round(float(wt[np.array([sets[k].startswith(pre) for k in tr])].sum()), 3) for pre in ['real-', 'proc-', 'ext-']}, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((args.epoch_size or len(tr)) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        model.train()
        perm = rng.choice(tr, size=args.epoch_size or len(tr), replace=True, p=wt) if args.weight or args.epoch_size else rng.permutation(tr)
        t0 = time.time(); tot = 0; n = 0
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            j = np.sort(perm[i:i + args.bs])
            x = augment(to_input(X[j], args.size, dev), gen, args.strong)
            s = torch.from_numpy(S[j]).to(dev)
            y = torch.from_numpy(Y[j]).to(dev)
            loss = F.cross_entropy(model(x, s), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(j); n += len(j)
        pv, _ = evaluate(model, X, S, Y, args.size, va, dev=dev)
        msg = f'epoch {ep + 1}/{args.epochs}: loss {tot / max(1, n):.4f}, validation {((pv == Y[va]).mean() * 100):.2f} %'
        real = np.array([sets[k].startswith('real-') for k in va])
        if real.any(): msg += f' (real {((pv == Y[va])[real].mean() * 100):.2f} %)'
        if len(ho):
            ph, _ = evaluate(model, X, S, Y, args.size, ho, dev=dev)
            msg += f', held-out sets {((ph == Y[ho]).mean() * 100):.2f} %'
        print(msg + f' ({time.time() - t0:.0f} s)', flush=True)
    # the model first, the reports after it
    if args.save:
        torch.save(model.state_dict(), args.save)
    if args.out:
        export(model, args.size, ns, args.out, {'version': f'n1-{args.size}-{time.strftime("%Y%m%d")}', 'note': f'{len(tr)} training squares, {args.epochs} epochs, input {args.size} px'})
        model.to(dev)   # export() moved it to the CPU
    pv, _ = evaluate(model, X, S, Y, args.size, va, dev=dev)
    report('validation', pv, Y[va], groups=[sets[k].split('-')[0] for k in va])
    if len(ho):
        ph, _ = evaluate(model, X, S, Y, args.size, ho, dev=dev)
        report('held-out sets', ph, Y[ho], groups=[sets[k] for k in ho])
    if args.test:
        XT, ST, YT, MT = load(args.test.split(','), args.size)
        pt, _ = evaluate(model, XT, ST, YT, args.size, dev=dev)
        report('test', pt, YT, groups=[m[3].split('-')[0] for m in MT])
        conf = np.zeros((len(CLASSES), len(CLASSES)), int)
        for a, b in zip(YT, pt): conf[a, b] += 1
        print('confusions (truth → read):', ', '.join(f'{CLASSES[a]}→{CLASSES[b]} {conf[a, b]}' for a, b in sorted(((a, b) for a in range(13) for b in range(13) if a != b and conf[a, b]), key=lambda ab: -conf[ab[0], ab[1]])[:24]))


if __name__ == '__main__':
    main()
