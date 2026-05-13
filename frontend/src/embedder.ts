/**
 * BGE Embedding — transformers.js (ort-web WASM)
 * 支持自动下载和手动本地文件两种加载方式
 */
import { pipeline, env, type FeatureExtractionPipeline } from '@xenova/transformers';

// 缓存目录名，用户手动下载后放到此目录
const MODEL_REPO = 'Xenova/bge-small-zh-v1.5';
const REQUIRED_FILES = [
  'tokenizer.json',
  'tokenizer_config.json',
  'onnx/model.onnx',
  'config.json',
];

let extractor: FeatureExtractionPipeline | null = null;
let modelLoaded = false;

export function getModelRepo(): string {
  return MODEL_REPO;
}

export function getRequiredFiles(): string[] {
  return REQUIRED_FILES;
}

/**
 * 从 HuggingFace 自动下载模型
 */
export async function initAuto(
  onProgress?: (msg: string) => void,
): Promise<void> {
  if (modelLoaded) return;
  onProgress?.('Downloading model from HuggingFace...');

  extractor = await pipeline('feature-extraction', MODEL_REPO, {
    progress_callback: (info: { status: string; file?: string; name?: string }) => {
      if (info.status === 'download' && info.file) {
        onProgress?.(`Downloading ${info.file}...`);
      } else if (info.status === 'progress') {
        onProgress?.('Loading weights...');
      } else if (info.status === 'done' && info.name) {
        onProgress?.(`Loaded ${info.name}`);
      }
    },
  });
  modelLoaded = true;
  onProgress?.('Model ready.');
}

/**
 * 从用户选择的本地文件加载模型
 * @param files - 用户选择的文件列表 (来自 <input webkitdirectory>)
 * @param onProgress - 进度回调
 */
export async function initFromFiles(
  files: File[],
  onProgress?: (msg: string) => void,
): Promise<void> {
  if (modelLoaded) return;

  // 构建 相对路径 → File 的映射
  const fileMap = new Map<string, File>();
  for (const f of files) {
    // webkitRelativePath 是相对于选中目录的路径
    const relPath = (f as any).webkitRelativePath || f.name;
    fileMap.set(relPath, f);
  }

  // 验证必需文件
  const missing: string[] = [];
  for (const required of REQUIRED_FILES) {
    if (!fileMap.has(required)) {
      missing.push(required);
    }
  }
  if (missing.length > 0) {
    throw new Error(`Missing required files:\n  ${missing.join('\n  ')}\n\nPlease download the model from:\nhttps://huggingface.co/${MODEL_REPO}`);
  }

  // 将文件读取为 ArrayBuffer 并创建 Blob URL
  onProgress?.('Reading local files...');
  const blobUrls = new Map<string, string>();
  for (const [path, file] of fileMap) {
    const url = URL.createObjectURL(file);
    blobUrls.set(path, url);
  }

  // 拦截 fetch，只用一次：让 transformers.js 从 Blob URL 加载
  const origFetch = globalThis.fetch.bind(globalThis);
  let fetchCount = 0;
  (globalThis as any).fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const urlStr = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

    // 检查是否匹配模型文件请求
    for (const [path, blobUrl] of blobUrls) {
      if (urlStr.includes(path) || urlStr.endsWith(path)) {
        fetchCount++;
        onProgress?.(`Loading ${path} (${fetchCount}/${REQUIRED_FILES.length})...`);
        const r = await fetch(blobUrl, init);
        if (!r.ok) throw new Error(`Failed to load ${path}`);
        return r;
      }
    }
    // 不匹配的请求走原始 fetch
    return origFetch(input, init);
  };

  try {
    onProgress?.('Initializing pipeline...');
    extractor = await pipeline('feature-extraction', MODEL_REPO, {
      local_files_only: true,
    });
    modelLoaded = true;
    onProgress?.('Model loaded from local files!');
  } finally {
    // 恢复原始 fetch
    globalThis.fetch = origFetch;
  }

  // 释放 Blob URL
  for (const url of blobUrls.values()) {
    URL.revokeObjectURL(url);
  }
}

export function isModelReady(): boolean {
  return modelLoaded;
}

export async function encode(texts: string[]): Promise<number[][]> {
  if (!extractor) throw new Error('Embedder not initialized');

  const embeddings: number[][] = [];
  for (const text of texts) {
    const output = await extractor(text, {
      pooling: 'mean',
      normalize: true,
    });
    embeddings.push(Array.from(output.data as Float32Array));
  }
  return embeddings;
}

export function cosineSim(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}
