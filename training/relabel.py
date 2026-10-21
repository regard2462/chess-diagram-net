# The readings of a newer network for squares already collected (scan-books.js shards): the stored squares are the
# network's input, so a model saved by train.py (--save) reads them again without the pages.
#   python training/relabel.py <model.pt> <prefix>[,<prefix>…] [--suffix l2] [--ch 16,32,48,64] [--hidden 96]
# Writes <prefix>.<suffix> (float32, 13 logits per square), for label.py sheets --logits <suffix>.
import argparse, json, os, sys
import numpy as np
sys.modules.setdefault('transformers', None)
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Net, to_input

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('prefixes')
ap.add_argument('--suffix', default='l2'); ap.add_argument('--ch', default='16,32,48,64'); ap.add_argument('--hidden', type=int, default=96)
ap.add_argument('--size', type=int, default=32)
a = ap.parse_args()
torch.set_num_threads(4)
model = None
for p in a.prefixes.split(','):
    info = json.load(open(p + '.json', encoding='utf8'))
    n = info['n']
    if model is None:
        model = Net(info['scalars'], a.size, [int(c) for c in a.ch.split(',')], a.hidden)
        model.load_state_dict(torch.load(a.model, map_location='cpu')); model.eval()
    if not n:
        np.zeros((0, 13), np.float32).tofile(p + '.' + a.suffix); continue
    X = np.fromfile(p + '.x', dtype=np.uint8).reshape(n, 2, info['N'], info['N'])
    S = np.fromfile(p + '.s', dtype=np.float32).reshape(n, info['scalars'])
    out = []
    with torch.no_grad():
        for i in range(0, n, 2048):
            out.append(model(to_input(X[i:i + 2048], a.size), torch.from_numpy(S[i:i + 2048])).numpy().astype(np.float32))
    np.concatenate(out).tofile(p + '.' + a.suffix)
print('relabelled', a.prefixes.count(',') + 1, 'shards')
