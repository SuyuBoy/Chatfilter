/**
 * Entry point — 纯前端弹幕语义聚类
 */
import { PipelineEngine } from './engine';
import { isModelReady } from './embedder';
import { initAuto, initFromFiles, getModelRepo, getRequiredFiles } from './embedder';

const engine = new PipelineEngine();

// ── Model Setup Modal ──
let modelReady = false;

function showSetupModal() {
  const overlay = document.getElementById('setup_overlay')!;
  overlay.style.display = 'flex';
}

function hideSetupModal() {
  const overlay = document.getElementById('setup_overlay')!;
  overlay.style.display = 'none';
}

function setStatus(msg: string, isError = false) {
  const el = document.getElementById('setup_status')!;
  el.textContent = msg;
  el.style.color = isError ? 'var(--red)' : 'var(--muted)';
}

function setProgress(msg: string) {
  const el = document.getElementById('setup_progress')!;
  el.style.display = msg ? 'block' : 'none';
  el.textContent = msg;
}

function setFileChecklist(files: string[], ok: string[] = []) {
  const el = document.getElementById('file_checklist')!;
  el.innerHTML = files.map(f => {
    const found = ok.some(o => o.endsWith(f));
    const icon = found ? '✅' : '❌';
    return `<div style="font-size:11px;padding:2px 0">${icon} ${f}${found ? '' : ''}</div>`;
  }).join('');
}

// Auto download
(document.getElementById('btn_auto') as HTMLButtonElement).onclick = async () => {
  const btn = document.getElementById('btn_auto') as HTMLButtonElement;
  const btnManual = document.getElementById('btn_manual') as HTMLButtonElement;
  btn.disabled = true;
  btnManual.disabled = true;
  setStatus('Starting download...');
  setProgress('Connecting to HuggingFace...');

  try {
    await initAuto((msg) => {
      setProgress(msg);
    });
    modelReady = true;
    setStatus('Model loaded successfully!');
    setTimeout(hideSetupModal, 1000);
    updateState();
  } catch (e: any) {
    setStatus(`Download failed: ${e.message}`, true);
    setProgress(`Tip: use manual mode — download from https://huggingface.co/${getModelRepo()}`);
    btn.disabled = false;
    btnManual.disabled = false;
  }
};

// Manual file selection
(document.getElementById('btn_manual') as HTMLButtonElement).onclick = () => {
  const picker = document.getElementById('dir_picker') as HTMLInputElement;
  const checklist = document.getElementById('file_checklist_area')!;
  checklist.style.display = 'block';
  setFileChecklist(getRequiredFiles());
  picker.click();
};

(document.getElementById('dir_picker') as HTMLInputElement).onchange = async () => {
  const picker = document.getElementById('dir_picker') as HTMLInputElement;
  const files = Array.from(picker.files || []);
  if (files.length === 0) return;

  // Show found files
  const foundPaths = files.map(f => (f as any).webkitRelativePath || f.name);
  setFileChecklist(getRequiredFiles(), foundPaths);
  setStatus(`Found ${files.length} files. Validating...`);
  setProgress('Reading files...');

  const btnAuto = document.getElementById('btn_auto') as HTMLButtonElement;
  const btnManual = document.getElementById('btn_manual') as HTMLButtonElement;
  btnAuto.disabled = true;
  btnManual.disabled = true;

  try {
    await initFromFiles(files, (msg) => {
      setProgress(msg);
    });
    modelReady = true;
    setStatus('Model loaded from local files!');
    setTimeout(hideSetupModal, 1000);
    updateState();
  } catch (e: any) {
    setStatus(e.message, true);
    setProgress('');
    btnAuto.disabled = false;
    btnManual.disabled = false;
  }
};

// Show modal on start
showSetupModal();

// ── UI State ──
let recentLog: Array<{ id: number; raw: string; canonical: string; clusterId: string; slotId: number }> = [];
let logSeq = 0;
let maxSlots = 40;

// ── Ingest ──
async function processIngest(text: string) {
  const result = await engine.ingest(text);
  if (!result.filtered) {
    recentLog.unshift({
      id: ++logSeq,
      raw: result.rawText,
      canonical: result.canonical,
      clusterId: result.clusterId,
      slotId: result.slotId,
    });
    if (recentLog.length > 200) recentLog.pop();
  }
  updateState();
}

function queueIngest(text: string) {
  if (modelReady) {
    processIngest(text);
  }
}

// ── Render ──
function updateState() {
  const s = engine.getState();
  maxSlots = s.maxSlots;

  document.getElementById('st_ingested')!.textContent = String(s.ingested);
  document.getElementById('st_unique')!.textContent = String(s.unique);
  document.getElementById('st_clusters')!.textContent = String(s.clusters.length);
  document.getElementById('st_slots')!.textContent = String(s.maxSlots);
  document.getElementById('st_ct')!.textContent = s.centroid.toFixed(2);
  document.getElementById('st_at')!.textContent = s.anchor.toFixed(2);
  document.getElementById('st_cache')!.textContent = (s.cacheHitRate * 100).toFixed(1) + '%';

  document.getElementById('cnt_raw')!.textContent = String(recentLog.length);
  renderRaw();
  document.getElementById('cnt_pre')!.textContent = String(recentLog.length);
  renderPre();
  document.getElementById('cnt_clu')!.textContent = String(s.clusters.length);
  renderPermanent(s.permanent);
  renderClusters(s.clusters);

  if (modelReady) {
    document.getElementById('model_status')!.textContent = 'Ready';
  }
}

function esc(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function heatColor(ratio: number): string {
  const logR = ratio > 0 ? Math.log(1 + ratio * 9) / Math.log(10) : 0;
  const r = Math.round(9 + logR * 241);
  const g = Math.round(105 - logR * 12);
  const b = Math.round(218 - logR * 100);
  return `rgba(${r},${g},${b},0.22)`;
}

// Raw column
let lastRawIds = new Set<string>();
let rawSeq = 0;
function renderRaw() {
  const el = document.getElementById('raw_list')!;
  const currentIds = new Set<string>();
  for (const item of recentLog) {
    const id = String(item.id);
    currentIds.add(id);
    if (!lastRawIds.has(id)) {
      const div = document.createElement('div');
      div.className = 'entry';
      div.innerHTML = `<span class="idx">${++rawSeq}</span><span class="txt">${esc(item.raw)}</span>`;
      el.prepend(div);
    }
  }
  lastRawIds = currentIds;
  while (el.children.length > 200) el.lastChild!.remove();
  el.scrollTop = 0;
}

// Preprocessed column
let lastPreIds = new Set<string>();
function renderPre() {
  const el = document.getElementById('pre_list')!;
  const currentIds = new Set<string>();
  for (const item of recentLog) {
    const id = String(item.id);
    currentIds.add(id);
    if (!lastPreIds.has(id)) {
      const changed = item.raw !== item.canonical;
      const div = document.createElement('div');
      div.className = 'entry' + (changed ? ' new' : '');
      const changedTag = changed ? '<span class="tag">归一化</span>' : '';
      div.innerHTML = `<span class="idx">&#8203;</span><span class="txt">${esc(item.canonical)}${changedTag}</span>`;
      el.prepend(div);
    }
  }
  lastPreIds = currentIds;
  while (el.children.length > 200) el.lastChild!.remove();
  el.scrollTop = 0;
}

// Permanent bar
function renderPermanent(perms: any[]) {
  const bar = document.getElementById('perm_bar')!;
  if (!perms || !perms.length) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';
  const maxC = Math.max(...perms.map((p: any) => p.totalCount || 0), 1);
  let html = '<span style="color:var(--purple);font-weight:600;flex-shrink:0;">🏛️ 热点</span>';
  for (const p of perms) {
    const ratio = (p.totalCount || 0) / maxC;
    const bg = heatColor(ratio);
    html += `<span style="background:${bg};color:var(--purple);padding:1px 6px;border-radius:3px;margin:0 2px;font-size:11px;white-space:nowrap;flex-shrink:0;">${esc((p.canonicalText || '').slice(0, 8))} <b>${p.totalCount || 0}</b></span>`;
  }
  html += `<span style="font-size:10px;color:var(--muted);flex-shrink:0;">${perms.length}个</span>`;
  bar.innerHTML = html;
}

// Cluster grid
let lastCluIds = new Set<number>();
function renderClusters(clusters: any[]) {
  const grid = document.getElementById('clu_grid')!;
  const currentIds = new Set<number>();
  const map: Record<number, any> = {};
  let maxCount = 1;
  for (const c of clusters) {
    map[c.slotId] = c;
    currentIds.add(c.slotId);
    if (c.totalCount > maxCount) maxCount = c.totalCount;
  }
  for (let sid = 1; sid <= maxSlots; sid++) {
    const c = map[sid];
    let card = grid.querySelector(`[data-sid="${sid}"]`) as HTMLElement;
    if (!card) {
      card = document.createElement('div');
      card.className = 'cluster-card';
      card.dataset.sid = String(sid);
      grid.appendChild(card);
    }
    if (c) {
      const latest = c.latestRaw || '';
      const topEx = (c.topExamples || [])[0] || '';
      const raw = latest || topEx;
      const ratio = (c.totalCount || 0) / maxCount;
      const bg = heatColor(ratio);
      card.innerHTML = `
        <div class="head">
          <span class="canonical">[${String(c.slotId).padStart(2, '0')}] ${esc(c.canonicalText)}</span>
          <span class="cnt">${c.totalCount || 0}次</span>
        </div>
        ${raw ? `<div class="members" style="margin-top:2px;"><span class="member">${esc(raw.slice(0, 20))}</span></div>` : ''}
      `;
      card.style.background = bg;
      card.style.opacity = '1';
    } else {
      card.innerHTML = `<div class="head"><span style="color:var(--muted)">[${String(sid).padStart(2, '0')}] —</span></div>`;
      card.style.opacity = '0.4';
    }
  }
  for (const child of [...grid.children]) {
    const sid = parseInt((child as HTMLElement).dataset.sid || '0');
    if (sid > maxSlots) child.remove();
  }
  lastCluIds = currentIds;
}

// ── Input handlers ──
document.getElementById('manual_send')!.addEventListener('click', () => {
  const inp = document.getElementById('manual_input') as HTMLInputElement;
  const text = inp.value.trim();
  if (!text) return;
  queueIngest(text);
  inp.value = '';
  inp.focus();
});

document.getElementById('manual_input')!.addEventListener('keydown', (e) => {
  if ((e as KeyboardEvent).key === 'Enter') {
    const inp = document.getElementById('manual_input') as HTMLInputElement;
    const text = inp.value.trim();
    if (!text) return;
    queueIngest(text);
    inp.value = '';
  }
});

document.getElementById('apply_threshold')!.addEventListener('click', () => {
  const ct = parseFloat((document.getElementById('in_ct') as HTMLInputElement).value) || 0.40;
  const at = parseFloat((document.getElementById('in_at') as HTMLInputElement).value) || 0.60;
  engine.cluster.config.centroidThreshold = ct;
  engine.cluster.config.anchorThreshold = at;
  updateState();
});

// Card width slider
document.getElementById('card_width')!.addEventListener('input', (e) => {
  const w = (e.target as HTMLInputElement).value;
  document.getElementById('card_width_val')!.textContent = w + 'px';
  document.getElementById('clu_grid')!.style.gridTemplateColumns = `repeat(auto-fill,minmax(${w}px,1fr))`;
});

// File bulk ingest
document.getElementById('file_input')!.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const text = await file.text();
  const lines = text.split(/[\n\r]+/).filter(l => l.trim());
  const bulkStatus = document.getElementById('bulk_status')!;
  bulkStatus.textContent = `Processing ${lines.length} messages...`;
  let i = 0;
  const batchSize = 10;
  async function nextBatch() {
    const batch = lines.slice(i, i + batchSize);
    if (batch.length === 0) {
      bulkStatus.textContent = `Done: ${lines.length} messages processed.`;
      return;
    }
    await engine.ingestBatch(batch.map(t => t.trim()));
    i += batchSize;
    updateState();
    bulkStatus.textContent = `Processed ${Math.min(i, lines.length)}/${lines.length}...`;
    requestAnimationFrame(() => nextBatch());
  }
  nextBatch();
});
