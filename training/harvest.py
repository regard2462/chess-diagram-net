# Collects the diagrams of a folder of chess books (PDF, read only) for the square classifier: the pages of every
# book are rendered in a shuffled order, in chunks, at a width of 800 to 1400 pixels each (as different screens show a
# page), and training/scan-books.js finds and analyses the boards on them, until a book has given --max
# diagrams or has no pages left. A book without a board in its first --probe pages is skipped.
#   python training/harvest.py <books folder> <out folder> [--max 250] [--probe 60] [--chunk 40] [--jobs 4]
#                                      [--only <substring>] [--low]
# Writes <out>/books.json (book id → file, pages, diagrams) and per book <out>/<id>/c<k>.{x,s,y,l,json}. The ids are
# hashes of the relative paths, so that the samples carry no titles. Needs PyMuPDF.
import ctypes, fitz, hashlib, json, os, random, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

args = sys.argv[1:]
root, out = args[0], args[1]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
MAX, PROBE, CHUNK, JOBS = opt('--max', 250), opt('--probe', 60), opt('--chunk', 40), opt('--jobs', 4)
ONLY = opt('--only', '')
LOW = '--low' in args
# --low: below normal priority for the driver and the scans, other work on this computer goes first
if LOW:
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception: pass
os.makedirs(out, exist_ok=True)
lock = threading.Lock()
books_file = os.path.join(out, 'books.json')
books = json.load(open(books_file, encoding='utf8')) if os.path.exists(books_file) else {}


def book_id(rel):
    return hashlib.sha1(rel.replace('\\', '/').encode('utf8')).hexdigest()[:10]


def render(doc, pages, folder, rnd):
    res = []
    for pn in pages:
        try:
            pg = doc[pn]
            W = rnd.randint(800, 1400)
            z = W / max(1, pg.rect.width)
            if pg.rect.height * z > 3000: z = 3000 / pg.rect.height
            pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
            f = f'p{pn:05d}.png'
            pix.save(os.path.join(folder, f))
            res.append({'page': pn, 'file': f, 'w': pix.width, 'h': pix.height, 'zoom': z})
        except Exception as e:
            print('  page', pn, 'failed:', str(e)[:100], flush=True)
    json.dump(res, open(os.path.join(folder, 'pages.json'), 'w'))
    return len(res)


def do_book(path):
    rel = os.path.relpath(path, root)
    bid = book_id(rel)
    with lock:
        if bid in books and books[bid].get('done'): return
    bdir = os.path.join(out, bid)
    os.makedirs(bdir, exist_ok=True)
    try:
        doc = fitz.open(path)
    except Exception as e:
        print('cannot open', bid, str(e)[:100], flush=True); return
    n = doc.page_count
    rnd = random.Random(int(bid, 16))
    order = list(range(n)); rnd.shuffle(order)
    total, k, t0 = 0, 0, time.time()
    while order and total < MAX:
        chunk, order = order[:CHUNK], order[CHUNK:]
        tmp = tempfile.mkdtemp(prefix='harvest-')
        try:
            render(doc, sorted(chunk), tmp, rnd)
            prefix = os.path.join(bdir, f'c{k}')
            r = subprocess.run(['node', '--no-warnings', os.path.join(HERE, 'scan-books.js'), tmp, prefix, '--book', bid, '--seed', str(rnd.randint(1, 10 ** 6))] + (['--low'] if LOW else []),
                               capture_output=True, text=True, encoding='utf8', errors='replace')
            if r.returncode != 0: print(bid, "scan failed", r.stderr[-300:], flush=True); break
            print("  ", r.stdout.strip()[-200:], flush=True)
            got = json.load(open(prefix + '.json', encoding='utf8'))
            total += len(got['diagrams'])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        k += 1
        if k * CHUNK >= PROBE and total == 0: break
    with lock:
        books[bid] = {'file': rel, 'pages': n, 'chunks': k, 'diagrams': total, 'seconds': round(time.time() - t0), 'done': True}
        json.dump(books, open(books_file, 'w', encoding='utf8'), indent=1, ensure_ascii=False)
    print(f'{bid} {total} diagrams from {min(n, k * CHUNK)} of {n} pages in {time.time() - t0:.0f} s', flush=True)


files = []
for dp, dn, fn in os.walk(root):
    for f in fn:
        if f.lower().endswith('.pdf') and ONLY.lower() in os.path.join(dp, f).lower():
            files.append(os.path.join(dp, f))
files.sort()
# the same book twice (same number of pages and the same first words of the name) is read once
seen, todo = {}, []
for p in files:
    try:
        with fitz.open(p) as d: key = (d.page_count, os.path.basename(p)[:18].lower())
    except Exception: key = (p,)
    if key in seen: continue
    seen[key] = p; todo.append(p)
print(len(files), 'PDFs,', len(todo), 'after duplicates', flush=True)
# a shuffled order (fixed): a run cut short still covers all kinds of books
random.Random(7).shuffle(todo)
with ThreadPoolExecutor(JOBS) as ex:
    list(ex.map(do_book, todo))
print('HARVEST DONE', sum(b['diagrams'] for b in books.values()), 'diagrams in', len(books), 'books')
