/**
 * 弹幕语义聚类 — 纯前端版 (transformers.js + WASM)
 * 预处理 → embedding → 聚类 → UI
 */

import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

// ═══════════════════════════════════════════════
//  1. 预处理 (preprocess)
// ═══════════════════════════════════════════════

// ── 分词 ──
const WORD_DICT = new Set([
  '哈哈哈','呵呵呵','嘿嘿嘿','啦啦啦','呜呜呜','弹幕','主播','加油',
  '厉害','牛逼','无敌','666','233','好看','好听','喜欢','可爱','搞笑',
  '笑死','太强','好帅','什么','怎么','为什么','来了','打卡','第一','前排',
]);
function segment(text: string): string[] {
  const words: string[] = [];
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (let len = Math.min(4, text.length - i); len >= 1; len--) {
      const sub = text.slice(i, i + len);
      if (WORD_DICT.has(sub)) { words.push(sub); i += len; matched = true; break; }
    }
    if (!matched) { words.push(text[i]); i++; }
  }
  return words;
}

// ── 拼音 ──
const PINYIN_MAP: Record<string, string> = {
  '哈':'ha','呵':'he','嘿':'hei','啦':'la','呜':'wu','帅':'shuai',
  '强':'qiang','牛':'niu','棒':'bang','好':'hao','爱':'ai','喜':'xi',
  '欢':'huan','笑':'xiao','哭':'ku','我':'wo','你':'ni','他':'ta','她':'ta',
  '是':'shi','的':'de','了':'le','在':'zai','有':'you','不':'bu','这':'zhe',
  '那':'na','会':'hui','能':'neng','要':'yao','说':'shuo','看':'kan','听':'ting',
  '来':'lai','去':'qu','上':'shang','下':'xia','大':'da','小':'xiao',
  '真':'zhen','快':'kuai','慢':'man','高':'gao','美':'mei','丑':'chou',
  '新':'xin','旧':'jiu','冷':'leng','热':'re','吃':'chi','喝':'he','玩':'wan',
  '跑':'pao','走':'zou','飞':'fei','跳':'tiao','游':'you','神':'shen','鬼':'gui',
  '死':'si','活':'huo','生':'sheng','歌':'ge','舞':'wu','唱':'chang',
  '弹':'tan','琴':'qin','书':'shu','画':'hua','弹幕':'danmu','主播':'zhubo',
};
function toPinyin(text: string): string {
  let r = ''; for (const ch of text) r += PINYIN_MAP[ch] || ch; return r;
}

const VARIANTS: Record<string, string> = {
  'xswl':'笑死我了','yyds':'永远的神','awsl':'啊我死了','u1s1':'有一说一',
  'srds':'虽然但是','zqsg':'真情实感','dbq':'对不起','nsdd':'你说得对',
  'tql':'太强了','sdl':'速度了',
};
function normalize(text: string): string {
  return VARIANTS[text.toLowerCase()] || text;
}

export function basicCleanse(text: string, minLen = 1, maxLen = 128): string | null {
  if (!text) return null;
  text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, '');
  text = text.normalize('NFKC');
  text = text.replace(/\s+/g, ' ').trim();
  if (!text) return null;
  if (/^\d+$/.test(text) && text.length > 4) return null;
  if (text.length < minLen || text.length > maxLen) return null;
  return text;
}

export function compressCycle(text: string): string {
  return text.replace(/(.+?)\1{2,}/g, '$1');
}

export function preprocess(rawText: string): { text: string; normalized: string } | null {
  const c = basicCleanse(rawText); if (!c) return null;
  const n = normalize(c); const r = compressCycle(n);
  return { text: r, normalized: r };
}

// ── SimHash ──
function hashString(s: string): bigint {
  let h = 5381n; for (let i = 0; i < s.length; i++) h = ((h << 5n) + h) ^ BigInt(s.charCodeAt(i));
  return h & ((1n << 64n) - 1n);
}
export function computeSimhash(tokens: string[], bits = 64): bigint {
  const vec = new Int32Array(bits);
  for (const token of tokens) {
    let h = hashString(token);
    for (let i = 0; i < bits; i++) { if ((h >> BigInt(i)) & 1n) vec[i]++; else vec[i]--; }
  }
  let fp = 0n; for (let i = 0; i < bits; i++) if (vec[i] > 0) fp |= (1n << BigInt(i));
  return fp;
}
export function hammingDistance(a: bigint, b: bigint): number {
  let xor = a ^ b, d = 0; while (xor > 0n) { d++; xor &= xor - 1n; } return d;
}

// ── DedupStore ──
export class DedupStore {
  private store = new Map<string, { count: number; raws: string[] }>();
  add(canonical: string, raw: string): boolean {
    const e = this.store.get(canonical);
    if (e) { e.count++; if (!e.raws.includes(raw)) e.raws.push(raw); return false; }
    this.store.set(canonical, { count: 1, raws: [raw] }); return true;
  }
  getCount(c: string) { return this.store.get(c)?.count ?? 0; }
  get size() { return this.store.size; }
  clear() { this.store.clear(); }
}


// ═══════════════════════════════════════════════
//  2. Embedding (embedder)
// ═══════════════════════════════════════════════

const MODEL_REPO = 'Xenova/bge-small-zh-v1.5';
const REQUIRED_FILES = ['tokenizer.json','tokenizer_config.json','onnx/model.onnx','config.json'];

let extractor: FeatureExtractionPipeline | null = null;
let modelLoaded = false;

export function getModelRepo() { return MODEL_REPO; }
export function getRequiredFiles() { return REQUIRED_FILES; }
export function isModelReady() { return modelLoaded; }

export async function initAuto(onProgress?: (m: string) => void): Promise<void> {
  if (modelLoaded) return;
  onProgress?.('Downloading from HuggingFace...');
  extractor = await pipeline('feature-extraction', MODEL_REPO, {
    progress_callback: (info: any) => {
      if (info.status === 'download' && info.file) onProgress?.(`Downloading ${info.file}...`);
      else if (info.status === 'progress') onProgress?.('Loading weights...');
    },
  });
  modelLoaded = true; onProgress?.('Model ready.');
}

export async function initFromFiles(files: File[], onProgress?: (m: string) => void): Promise<void> {
  if (modelLoaded) return;
  const fileMap = new Map<string, File>();
  for (const f of files) fileMap.set((f as any).webkitRelativePath || f.name, f);
  const missing = REQUIRED_FILES.filter(r => !fileMap.has(r));
  if (missing.length) throw new Error(`Missing: ${missing.join(', ')}\nDownload: https://huggingface.co/${MODEL_REPO}`);
  onProgress?.('Reading local files...');
  const blobs = new Map<string, string>();
  for (const [p, f] of fileMap) blobs.set(p, URL.createObjectURL(f));
  const orig = globalThis.fetch.bind(globalThis); let n = 0;
  (globalThis as any).fetch = async (input: any, init?: any) => {
    const u = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    for (const [p, url] of blobs) {
      if (u.includes(p) || u.endsWith(p)) { n++; onProgress?.(`Loading ${p} (${n}/${REQUIRED_FILES.length})...`); return orig(url, init); }
    }
    return orig(input, init);
  };
  try {
    onProgress?.('Initializing pipeline...');
    extractor = await pipeline('feature-extraction', MODEL_REPO, { local_files_only: true });
    modelLoaded = true; onProgress?.('Model loaded from local files!');
  } finally {
    globalThis.fetch = orig;
    for (const url of blobs.values()) URL.revokeObjectURL(url);
  }
}

export async function encode(texts: string[]): Promise<number[][]> {
  if (!extractor) throw new Error('Not initialized');
  const r: number[][] = [];
  for (const t of texts) r.push(Array.from((await extractor(t, { pooling:'mean', normalize:true })).data as Float32Array));
  return r;
}

export function cosineSim(a: number[], b: number[]): number {
  let d=0, na=0, nb=0;
  for (let i = 0; i < a.length; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}


// ═══════════════════════════════════════════════
//  3. 聚类引擎 (cluster)
// ═══════════════════════════════════════════════

function lenPenalty(newLen: number, anchorAvg: number): number {
  const r = Math.max(newLen, anchorAvg) / Math.max(Math.min(newLen, anchorAvg), 1);
  if (r <= 1.5) return 1.0; if (r <= 3.0) return 0.9; return 0.8;
}

export interface SlotState {
  slotId: number; clusterId: string; canonicalText: string;
  totalCount: number; topExamples: string[]; latestRaw: string;
  lastUpdate: number; centroid: number[] | null; memberEmbeddings: number[][];
}

export interface ClusterConfig {
  centroidThreshold: number; anchorThreshold: number; maxSlots: number;
  mergeThreshold: number; splitVarianceThreshold: number; maintenanceInterval: number;
  permanentThreshold: number; permanentCentroidThreshold: number;
  permanentAnchorThreshold: number; permanentMergeThreshold: number;
  permanentSplitVarianceThreshold: number; permanentTtlSeconds: number;
  maxMemberEmbeddings: number;
}

export const DEFAULT_CLUSTER_CONFIG: ClusterConfig = {
  centroidThreshold: 0.4, anchorThreshold: 0.6, maxSlots: 40,
  mergeThreshold: 0.92, splitVarianceThreshold: 0.50, maintenanceInterval: 300,
  permanentThreshold: 100, permanentCentroidThreshold: 0.75, permanentAnchorThreshold: 0.82,
  permanentMergeThreshold: 0.94, permanentSplitVarianceThreshold: 0.45,
  permanentTtlSeconds: 600, maxMemberEmbeddings: 50,
};

function mkSlot(sid: number, cid: string, can: string, emb: number[], raw: string): SlotState {
  return { slotId:sid, clusterId:cid, canonicalText:can, totalCount:1, topExamples:[raw],
    latestRaw:raw, lastUpdate:Date.now(), centroid:[...emb], memberEmbeddings:[[...emb]] };
}

function internalSim(s: SlotState): number {
  if (s.memberEmbeddings.length < 2 || !s.centroid) return 1;
  return s.memberEmbeddings.reduce((a, e) => a + cosineSim(e, s.centroid!), 0) / s.memberEmbeddings.length;
}

function canJoin(s: SlotState, emb: number[], tlen: number, cth: number, ath: number): boolean {
  if (!s.centroid) return true;
  const sc = cosineSim(emb, s.centroid);
  if (sc < cth || sc < ath) return false;
  if (tlen > 0) { const p = lenPenalty(tlen, s.canonicalText.length || 1); if (sc * p < cth) return false; }
  return true;
}

export class ClusterEngine {
  config: ClusterConfig;
  slots = new Map<string, SlotState>();
  permanent = new Map<string, SlotState>();
  private npid = 1;
  totalIngested = 0;

  constructor(cfg: Partial<ClusterConfig> = {}) { this.config = { ...DEFAULT_CLUSTER_CONFIG, ...cfg }; }

  findCluster(emb: number[], tlen: number): { slot: SlotState; isPerm: boolean } | null {
    let r = this._find(this.permanent, emb, tlen, this.config.permanentCentroidThreshold, this.config.permanentAnchorThreshold, this.config.permanentSplitVarianceThreshold);
    if (r) return { slot: r, isPerm: true };
    r = this._find(this.slots, emb, tlen, this.config.centroidThreshold, this.config.anchorThreshold, this.config.splitVarianceThreshold);
    return r ? { slot: r, isPerm: false } : null;
  }

  private _find(d: Map<string, SlotState>, emb: number[], tlen: number, cth: number, ath: number, sth: number): SlotState | null {
    let best = -1, bestS: SlotState | null = null;
    for (const s of d.values()) {
      if (!s.centroid || !canJoin(s, emb, tlen, cth, ath)) continue;
      if (s.memberEmbeddings.length >= 3 && internalSim(s) < sth) continue;
      const sim = cosineSim(emb, s.centroid);
      if (sim > best) { best = sim; bestS = s; }
    }
    return bestS;
  }

  join(s: SlotState, emb: number[], raw: string, can: string) {
    s.totalCount++; s.latestRaw = raw; s.lastUpdate = Date.now();
    const o = Math.max(s.totalCount - 1, 1);
    if (s.centroid) s.centroid = s.centroid.map((c, i) => (c * o + emb[i]) / s.totalCount);
    if (!s.topExamples.includes(raw)) s.topExamples.push(raw);
    s.topExamples = [...new Set(s.topExamples)].sort((a, b) => b.length - a.length).slice(0, 3);
    this._addEmb(s, emb);
  }

  newSlot(can: string, emb: number[], raw: string): { slotId: number; clusterId: string } {
    const sid = this._alloc(); const cid = `c${sid}`;
    this.slots.set(can, mkSlot(sid, cid, can, emb, raw));
    return { slotId: sid, clusterId: cid };
  }

  private _alloc(): number {
    const max = this.config.maxSlots;
    if (this.slots.size < max) {
      const used = new Set([...this.slots.values()].map(s => s.slotId));
      for (let i = 1; i <= max; i++) if (!used.has(i)) return i;
      return this.slots.size + 1;
    }
    let old: SlotState | null = null;
    for (const s of this.slots.values()) if (!old || s.lastUpdate < old.lastUpdate) old = s;
    if (old) { this.slots.delete(old.canonicalText); return old.slotId; }
    return 1;
  }

  promote(s: SlotState) {
    const c = s.canonicalText; if (!this.slots.has(c)) return;
    this.slots.delete(c); s.slotId = this.npid; s.clusterId = `p${this.npid}`;
    this.npid++; this.permanent.set(c, s);
  }

  maintenance() {
    if (this.totalIngested % this.config.maintenanceInterval !== 0) return;
    this._merge(this.slots, this.config.mergeThreshold);
    this._split(this.slots, this.config.splitVarianceThreshold, this.config.maxSlots);
    this._merge(this.permanent, this.config.permanentMergeThreshold);
    this._split(this.permanent, this.config.permanentSplitVarianceThreshold, 9999);
    const now = Date.now();
    for (const [k, s] of this.permanent) if ((now - s.lastUpdate) / 1000 > this.config.permanentTtlSeconds) this.permanent.delete(k);
  }

  private _merge(d: Map<string, SlotState>, th: number) {
    const es = [...d.entries()]; const merged = new Set<string>();
    for (let i = 0; i < es.length; i++) {
      const [ci, si] = es[i]; if (merged.has(ci)) continue;
      for (let j = i + 1; j < es.length; j++) {
        const [cj, sj] = es[j]; if (merged.has(cj) || !si.centroid || !sj.centroid) continue;
        if (cosineSim(si.centroid, sj.centroid) > th) {
          if (si.totalCount >= sj.totalCount) { this._absorb(si, sj); merged.add(cj); }
          else { this._absorb(sj, si); merged.add(ci); }
        }
      }
    }
    for (const c of merged) d.delete(c);
  }

  private _absorb(t: SlotState, s: SlotState) {
    const n = t.totalCount + s.totalCount;
    if (t.centroid && s.centroid) t.centroid = t.centroid.map((c, i) => (c * t.totalCount + s.centroid![i] * s.totalCount) / n);
    t.totalCount = n; t.lastUpdate = Math.max(t.lastUpdate, s.lastUpdate);
    for (const e of s.topExamples) if (!t.topExamples.includes(e)) t.topExamples.push(e);
    t.topExamples = [...new Set(t.topExamples)].sort((a, b) => b.length - a.length).slice(0, 3);
    for (const e of s.memberEmbeddings) this._addEmb(t, e);
  }

  private _split(d: Map<string, SlotState>, th: number, max: number) {
    const ts: [string, SlotState][] = [];
    for (const [c, s] of d) { if (s.memberEmbeddings.length >= 4 && internalSim(s) < th) ts.push([c, s]); }
    for (const [c, s] of ts) {
      if (!d.has(c) || s.memberEmbeddings.length < 4) continue;
      const lbs = _kmeans2(s.memberEmbeddings);
      const g0 = s.memberEmbeddings.filter((_, k) => lbs[k] === 0);
      const g1 = s.memberEmbeddings.filter((_, k) => lbs[k] === 1);
      if (g0.length < 2 || g1.length < 2) continue;
      let nid: number;
      if (max < 9999) { nid = 0; const used = new Set([...d.values()].map(x => x.slotId)); for (let i = 1; i <= max; i++) if (!used.has(i)) { nid = i; break; } if (!nid) continue; }
      else nid = Math.max(...[...d.values()].map(x => x.slotId)) + 1;
      s.centroid = _norm(_mean(g0)); s.totalCount = g0.length; s.memberEmbeddings = g0;
      const pf = max === 9999 ? 'p' : 'c';
      d.set(`${s.canonicalText}*`, { slotId:nid, clusterId:`${pf}${nid}`, canonicalText:s.canonicalText, totalCount:g1.length, topExamples:s.topExamples.slice(0,1), latestRaw:s.latestRaw, lastUpdate:s.lastUpdate, centroid:_norm(_mean(g1)), memberEmbeddings:g1 });
    }
  }

  private _addEmb(s: SlotState, emb: number[]) {
    s.memberEmbeddings.push([...emb]);
    if (s.memberEmbeddings.length > this.config.maxMemberEmbeddings) {
      const h = Math.floor(this.config.maxMemberEmbeddings / 2);
      s.memberEmbeddings = [...s.memberEmbeddings.slice(0, h), ...s.memberEmbeddings.slice(-h)];
    }
  }

  getClusters() { return [...this.slots.values()].sort((a, b) => b.totalCount - a.totalCount); }
  getPermanent() { return [...this.permanent.values()].sort((a, b) => b.totalCount - a.totalCount); }
}

function _mean(vs: number[][]): number[] {
  const d = vs[0].length, m = new Array(d).fill(0);
  for (const v of vs) for (let i = 0; i < d; i++) m[i] += v[i];
  for (let i = 0; i < d; i++) m[i] /= vs.length;
  return m;
}
function _norm(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) + 1e-8; return v.map(x => x / n);
}
function _kmeans2(vs: number[][]): number[] {
  const n = vs.length; if (n < 2) return new Array(n).fill(0);
  let c0 = [...vs[0]], c1 = [...vs[Math.min(1, n - 1)]]; const lbs = new Array(n).fill(0);
  for (let it = 0; it < 10; it++) {
    let ch = false;
    for (let i = 0; i < n; i++) { const nl = cosineSim(vs[i], c0) >= cosineSim(vs[i], c1) ? 0 : 1; if (nl !== lbs[i]) { ch = true; lbs[i] = nl; } }
    if (!ch) break;
    const g0 = vs.filter((_, i) => lbs[i] === 0), g1 = vs.filter((_, i) => lbs[i] === 1);
    if (g0.length) c0 = _mean(g0); if (g1.length) c1 = _mean(g1);
  }
  return lbs;
}


// ═══════════════════════════════════════════════
//  4. 编排层 (engine)
// ═══════════════════════════════════════════════

export interface IngestResult {
  clusterId: string; canonical: string; rawText: string;
  slotId: number; isNew: boolean; filtered: boolean; permanent: boolean;
}
export interface PipelineState {
  ingested: number; unique: number; centroid: number; anchor: number;
  maxSlots: number; clusters: SlotState[]; permanent: SlotState[];
  cacheHitRate: number; modelReady: boolean;
}

export class PipelineEngine {
  cluster = new ClusterEngine();
  dedup = new DedupStore();
  cacheHits = 0; cacheTotal = 0; totalIngested = 0;

  async ingest(rawText: string): Promise<IngestResult> {
    this.totalIngested++; this.cluster.totalIngested = this.totalIngested;
    const pp = preprocess(rawText);
    if (!pp) return { clusterId:'', canonical:'', rawText, slotId:0, isNew:false, filtered:true, permanent:false };
    this.cacheTotal++;
    if (!this.dedup.add(pp.normalized, rawText)) this.cacheHits++;
    if (!isModelReady()) return { clusterId:'', canonical:pp.normalized, rawText, slotId:0, isNew:false, filtered:false, permanent:false };
    const [emb] = await encode([pp.normalized]);
    return this._cluster(emb, pp.normalized, rawText);
  }

  async ingestBatch(texts: string[]): Promise<IngestResult[]> {
    const out: IngestResult[] = [], need: { idx:number; can:string }[] = [], pre: { idx:number; can:string; raw:string }[] = [];
    for (let i = 0; i < texts.length; i++) {
      this.totalIngested++;
      const pp = preprocess(texts[i]);
      if (!pp) { out[i] = { clusterId:'', canonical:'', rawText:texts[i], slotId:0, isNew:false, filtered:true, permanent:false }; continue; }
      this.cacheTotal++; if (!this.dedup.add(pp.normalized, texts[i])) this.cacheHits++;
      pre.push({ idx:i, can:pp.normalized, raw:texts[i] }); need.push({ idx:i, can:pp.normalized });
    }
    this.cluster.totalIngested = this.totalIngested;
    if (need.length && isModelReady()) {
      const embs = await encode(need.map(n => n.can));
      const em = new Map<string, number[]>();
      for (let i = 0; i < need.length; i++) em.set(need[i].can, embs[i]);
      for (const { idx, can, raw } of pre) {
        const emb = em.get(can);
        out[idx] = emb ? this._cluster(emb, can, raw) : { clusterId:'', canonical:can, rawText:raw, slotId:0, isNew:false, filtered:false, permanent:false };
      }
      this.cluster.maintenance();
    }
    return out;
  }

  private _cluster(emb: number[], can: string, raw: string): IngestResult {
    const r = this.cluster.findCluster(emb, raw.length);
    if (r) {
      this.cluster.join(r.slot, emb, raw, can);
      if (!r.isPerm && r.slot.totalCount >= this.cluster.config.permanentThreshold) this.cluster.promote(r.slot);
      this.cluster.maintenance();
      return { clusterId:r.slot.clusterId, canonical:can, rawText:raw, slotId:r.slot.slotId, isNew:false, filtered:false, permanent:r.isPerm };
    }
    const { slotId, clusterId } = this.cluster.newSlot(can, emb, raw);
    this.cluster.maintenance();
    return { clusterId, canonical:can, rawText:raw, slotId, isNew:true, filtered:false, permanent:false };
  }

  getState(): PipelineState {
    return {
      ingested: this.totalIngested, unique: this.dedup.size,
      centroid: this.cluster.config.centroidThreshold, anchor: this.cluster.config.anchorThreshold,
      maxSlots: this.cluster.config.maxSlots, clusters: this.cluster.getClusters(),
      permanent: this.cluster.getPermanent(),
      cacheHitRate: this.cacheTotal > 0 ? this.cacheHits / this.cacheTotal : 0,
      modelReady: isModelReady(),
    };
  }
}


// ═══════════════════════════════════════════════
//  5. UI + Entry (main)
// ═══════════════════════════════════════════════

const engine = new PipelineEngine();
let modelReady = false;

// ── Setup Modal ──
function showModal() { document.getElementById('setup_overlay')!.style.display = 'flex'; }
function hideModal() { document.getElementById('setup_overlay')!.style.display = 'none'; }
function setStatus(m: string, err = false) {
  const el = document.getElementById('setup_status')!; el.textContent = m; el.style.color = err ? 'var(--red)' : 'var(--muted)';
}
function setProgress(m: string) {
  const el = document.getElementById('setup_progress')!; el.style.display = m ? 'block' : 'none'; el.textContent = m;
}
function setChecklist(files: string[], ok: string[] = []) {
  document.getElementById('file_checklist')!.innerHTML = files.map(f => {
    const found = ok.some(o => o.endsWith(f));
    return `<div style="font-size:11px;padding:2px 0">${found ? '✅' : '❌'} ${f}</div>`;
  }).join('');
}

document.getElementById('btn_auto')!.onclick = async () => {
  (document.getElementById('btn_auto') as HTMLButtonElement).disabled = true;
  (document.getElementById('btn_manual') as HTMLButtonElement).disabled = true;
  setStatus('Connecting...'); setProgress('');
  try {
    await initAuto(m => setProgress(m));
    modelReady = true; setStatus('Model ready!'); setTimeout(hideModal, 1000); updateState();
  } catch (e: any) {
    setStatus(`Failed: ${e.message}`, true);
    setProgress(`Tip: use manual mode — download from https://huggingface.co/${getModelRepo()}`);
    (document.getElementById('btn_auto') as HTMLButtonElement).disabled = false;
    (document.getElementById('btn_manual') as HTMLButtonElement).disabled = false;
  }
};

document.getElementById('btn_manual')!.onclick = () => {
  document.getElementById('file_checklist_area')!.style.display = 'block';
  setChecklist(getRequiredFiles());
  (document.getElementById('dir_picker') as HTMLInputElement).click();
};

document.getElementById('dir_picker')!.onchange = async () => {
  const fs = Array.from((document.getElementById('dir_picker') as HTMLInputElement).files || []);
  if (!fs.length) return;
  setChecklist(getRequiredFiles(), fs.map(f => (f as any).webkitRelativePath || f.name));
  setStatus(`Found ${fs.length} files. Validating...`);
  setProgress('Reading files...');
  (document.getElementById('btn_auto') as HTMLButtonElement).disabled = true;
  (document.getElementById('btn_manual') as HTMLButtonElement).disabled = true;
  try {
    await initFromFiles(fs, m => setProgress(m));
    modelReady = true; setStatus('Model loaded!'); setTimeout(hideModal, 1000); updateState();
  } catch (e: any) {
    setStatus(e.message, true); setProgress('');
    (document.getElementById('btn_auto') as HTMLButtonElement).disabled = false;
    (document.getElementById('btn_manual') as HTMLButtonElement).disabled = false;
  }
};

showModal();

// ── UI State ──
let recentLog: { id:number; raw:string; canonical:string; clusterId:string; slotId:number }[] = [];
let logSeq = 0, maxSlots = 40;

async function processIngest(text: string) {
  const r = await engine.ingest(text);
  if (!r.filtered) {
    recentLog.unshift({ id:++logSeq, raw:r.rawText, canonical:r.canonical, clusterId:r.clusterId, slotId:r.slotId });
    if (recentLog.length > 200) recentLog.pop();
  }
  updateState();
}
function queueIngest(text: string) { if (modelReady) processIngest(text); }

let lastRawIds = new Set<string>(), rawSeq = 0, lastPreIds = new Set<string>(), lastCluIds = new Set<number>();

function updateState() {
  const s = engine.getState(); maxSlots = s.maxSlots;
  document.getElementById('st_ingested')!.textContent = String(s.ingested);
  document.getElementById('st_unique')!.textContent = String(s.unique);
  document.getElementById('st_clusters')!.textContent = String(s.clusters.length);
  document.getElementById('st_slots')!.textContent = String(s.maxSlots);
  document.getElementById('st_ct')!.textContent = s.centroid.toFixed(2);
  document.getElementById('st_at')!.textContent = s.anchor.toFixed(2);
  document.getElementById('st_cache')!.textContent = (s.cacheHitRate * 100).toFixed(1) + '%';
  document.getElementById('cnt_raw')!.textContent = String(recentLog.length); renderRaw();
  document.getElementById('cnt_pre')!.textContent = String(recentLog.length); renderPre();
  document.getElementById('cnt_clu')!.textContent = String(s.clusters.length); renderPerm(s.permanent); renderClusters(s.clusters);
  if (modelReady) document.getElementById('model_status')!.textContent = 'Ready';
}

function e(s: string): string { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function hc(ratio: number): string {
  const lr = ratio > 0 ? Math.log(1 + ratio * 9) / Math.log(10) : 0;
  return `rgba(${Math.round(9 + lr * 241)},${Math.round(105 - lr * 12)},${Math.round(218 - lr * 100)},0.22)`;
}

function renderRaw() {
  const el = document.getElementById('raw_list')!, cur = new Set<string>();
  for (const item of recentLog) { const id = String(item.id); cur.add(id);
    if (!lastRawIds.has(id)) { const d = document.createElement('div'); d.className = 'entry'; d.innerHTML = `<span class="idx">${++rawSeq}</span><span class="txt">${e(item.raw)}</span>`; el.prepend(d); }
  }
  lastRawIds = cur; while (el.children.length > 200) el.lastChild!.remove(); el.scrollTop = 0;
}
function renderPre() {
  const el = document.getElementById('pre_list')!, cur = new Set<string>();
  for (const item of recentLog) { const id = String(item.id); cur.add(id);
    if (!lastPreIds.has(id)) { const ch = item.raw !== item.canonical; const d = document.createElement('div'); d.className = 'entry' + (ch ? ' new' : ''); d.innerHTML = `<span class="idx">&#8203;</span><span class="txt">${e(item.canonical)}${ch ? '<span class="tag">归一化</span>' : ''}</span>`; el.prepend(d); }
  }
  lastPreIds = cur; while (el.children.length > 200) el.lastChild!.remove(); el.scrollTop = 0;
}
function renderPerm(perms: any[]) {
  const bar = document.getElementById('perm_bar')!;
  if (!perms.length) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex'; const maxC = Math.max(...perms.map((p: any) => p.totalCount || 0), 1);
  let h = '<span style="color:var(--purple);font-weight:600;flex-shrink:0;">🏛️ 热点</span>';
  for (const p of perms) { const r = (p.totalCount || 0) / maxC, bg = hc(r); h += `<span style="background:${bg};color:var(--purple);padding:1px 6px;border-radius:3px;margin:0 2px;font-size:11px;white-space:nowrap;flex-shrink:0;">${e((p.canonicalText || '').slice(0, 8))} <b>${p.totalCount || 0}</b></span>`; }
  h += `<span style="font-size:10px;color:var(--muted);flex-shrink:0;">${perms.length}个</span>`; bar.innerHTML = h;
}
function renderClusters(clusters: any[]) {
  const grid = document.getElementById('clu_grid')!, cur = new Set<number>(), map: Record<number, any> = {};
  let maxC = 1; for (const c of clusters) { map[c.slotId] = c; cur.add(c.slotId); if (c.totalCount > maxC) maxC = c.totalCount; }
  for (let sid = 1; sid <= maxSlots; sid++) {
    const c = map[sid]; let card = grid.querySelector(`[data-sid="${sid}"]`) as HTMLElement;
    if (!card) { card = document.createElement('div'); card.className = 'cluster-card'; card.dataset.sid = String(sid); grid.appendChild(card); }
    if (c) { const raw = c.latestRaw || (c.topExamples || [])[0] || ''; const r = (c.totalCount || 0) / maxC, bg = hc(r);
      card.innerHTML = `<div class="head"><span class="canonical">[${String(c.slotId).padStart(2, '0')}] ${e(c.canonicalText)}</span><span class="cnt">${c.totalCount || 0}次</span></div>${raw ? `<div class="members" style="margin-top:2px;"><span class="member">${e(raw.slice(0, 20))}</span></div>` : ''}`;
      card.style.background = bg; card.style.opacity = '1'; }
    else { card.innerHTML = `<div class="head"><span style="color:var(--muted)">[${String(sid).padStart(2, '0')}] —</span></div>`; card.style.opacity = '0.4'; }
  }
  for (const ch of [...grid.children]) { const s = parseInt((ch as HTMLElement).dataset.sid || '0'); if (s > maxSlots) ch.remove(); }
  lastCluIds = cur;
}

document.getElementById('manual_send')!.onclick = () => { const i = document.getElementById('manual_input') as HTMLInputElement; const t = i.value.trim(); if (t) { queueIngest(t); i.value = ''; i.focus(); } };
document.getElementById('manual_input')!.onkeydown = (e) => { if ((e as KeyboardEvent).key === 'Enter') { const i = document.getElementById('manual_input') as HTMLInputElement; const t = i.value.trim(); if (t) { queueIngest(t); i.value = ''; } } };
document.getElementById('apply_threshold')!.onclick = () => { engine.cluster.config.centroidThreshold = parseFloat((document.getElementById('in_ct') as HTMLInputElement).value) || 0.4; engine.cluster.config.anchorThreshold = parseFloat((document.getElementById('in_at') as HTMLInputElement).value) || 0.6; updateState(); };
document.getElementById('card_width')!.oninput = (e) => { const w = (e.target as HTMLInputElement).value; document.getElementById('card_width_val')!.textContent = w + 'px'; document.getElementById('clu_grid')!.style.gridTemplateColumns = `repeat(auto-fill,minmax(${w}px,1fr))`; };
document.getElementById('file_input')!.onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0]; if (!f) return;
  const t = await f.text(), lines = t.split(/[\n\r]+/).filter(l => l.trim()), st = document.getElementById('bulk_status')!;
  st.textContent = `Processing ${lines.length} messages...`; let i = 0; const bs = 10;
  async function nb() { const b = lines.slice(i, i + bs); if (!b.length) { st.textContent = `Done: ${lines.length} msgs.`; return; } await engine.ingestBatch(b.map(l => l.trim())); i += bs; updateState(); st.textContent = `Processed ${Math.min(i, lines.length)}/${lines.length}...`; requestAnimationFrame(() => nb()); }
  nb();
};
