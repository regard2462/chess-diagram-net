// Corrections of recognised diagrams, remembered per book as templates (IndexedDB in the app's data folder):
// a book prints all its diagrams with one piece set, so the squares the reader fixed by hand teach the
// recognition that set. Entries: { role ('k'…'p' or '.' for "no piece"), black, light, feat (numbers), at }.
// The same database caches the templates of this computer (templates.js), so that only the first snip after an
// update has to build them.
const DB = 'lesesaal-diagrams', STORE = 'books', CACHE = 'templates';
const PER_CLASS = 24, MAX = 400;

let dbPromise = null;
function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, 2);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
        if (!req.result.objectStoreNames.contains(CACHE)) req.result.createObjectStore(CACHE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }).catch((e) => { dbPromise = null; throw e; });
  }
  return dbPromise;
}

const tx = async (mode, fn) => {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(out?.result ?? out);
    t.onerror = () => reject(t.error);
  });
};

/** the remembered templates of a book ([] without any) */
export async function loadBookTemplates(bookId) {
  if (!bookId) return [];
  try {
    const db = await open();
    return await new Promise((resolve) => {
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(String(bookId));
      req.onsuccess = () => resolve(Array.isArray(req.result) ? req.result : []);
      req.onerror = () => resolve([]);
    });
  } catch { return []; }
}

/** adds corrections; per class only the newest ones are kept */
export async function addBookTemplates(bookId, entries) {
  if (!bookId || !entries?.length) return 0;
  const old = await loadBookTemplates(bookId);
  const all = [...old, ...entries.map((e) => ({ role: e.role, black: !!e.black, light: !!e.light, feat: Array.from(e.feat, (v) => +v.toFixed(4)), at: Date.now() }))];
  const byClass = new Map();
  for (const e of all) { const k = e.role + (e.black ? 'b' : 'w'); const arr = byClass.get(k) || []; arr.push(e); byClass.set(k, arr); }
  let kept = [];
  for (const arr of byClass.values()) kept.push(...arr.slice(-PER_CLASS));
  kept = kept.sort((a, b) => a.at - b.at).slice(-MAX);
  await tx('readwrite', (s) => s.put(kept, String(bookId)));
  return kept.length;
}

export async function clearBookTemplates(bookId) {
  if (!bookId) return;
  await tx('readwrite', (s) => s.delete(String(bookId)));
}

/** cached template items under `key` (null without) */
export async function loadCachedTemplates(key) {
  try {
    const db = await open();
    return await new Promise((resolve) => {
      const req = db.transaction(CACHE, 'readonly').objectStore(CACHE).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch { return null; }
}

/** stores template items under `key`, replacing all older entries */
export async function storeCachedTemplates(key, items) {
  try {
    const db = await open();
    await new Promise((resolve, reject) => {
      const t = db.transaction(CACHE, 'readwrite');
      const st = t.objectStore(CACHE);
      st.clear();
      st.put(items, key);
      t.oncomplete = resolve; t.onerror = () => reject(t.error);
    });
  } catch (e) { console.warn('[chess] template cache not written', e); }
}
