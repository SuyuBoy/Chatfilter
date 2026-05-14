/**
 * 弹幕预处理 — 清洗、归一化、循环压缩、去重
 */

// 简单中文分词: 基于词典 + 单字切分
const WORD_DICT = new Set([
  '哈哈哈', '呵呵呵', '嘿嘿嘿', '啦啦啦', '呜呜呜',
  '弹幕', '主播', '加油', '厉害', '牛逼', '无敌', '666', '233',
  '好看', '好听', '喜欢', '可爱', '搞笑', '笑死', '太强', '好帅',
  '什么', '怎么', '为什么', '不是', '真的', '确实', '哈哈哈',
  '来了', '打卡', '第一', '前排', '有人吗', '没人',
]);

function segment(text: string): string[] {
  const words: string[] = [];
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (let len = Math.min(4, text.length - i); len >= 1; len--) {
      const sub = text.slice(i, i + len);
      if (WORD_DICT.has(sub)) {
        words.push(sub);
        i += len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      words.push(text[i]);
      i++;
    }
  }
  return words;
}

// 简单拼音映射 (常用字)
const PINYIN_MAP: Record<string, string> = {
  '哈': 'ha', '呵': 'he', '嘿': 'hei', '啦': 'la', '呜': 'wu',
  '帅': 'shuai', '强': 'qiang', '牛': 'niu', '棒': 'bang', '好': 'hao',
  '爱': 'ai', '喜': 'xi', '欢': 'huan', '笑': 'xiao', '哭': 'ku',
  '我': 'wo', '你': 'ni', '他': 'ta', '她': 'ta', '它': 'ta',
  '是': 'shi', '的': 'de', '了': 'le', '在': 'zai', '有': 'you',
  '不': 'bu', '这': 'zhe', '那': 'na', '会': 'hui', '能': 'neng',
  '要': 'yao', '说': 'shuo', '看': 'kan', '听': 'ting', '想': 'xiang',
  '来': 'lai', '去': 'qu', '上': 'shang', '下': 'xia', '大': 'da',
  '小': 'xiao', '多': 'duo', '少': 'shao', '真': 'zhen', '假': 'jia',
  '快': 'kuai', '慢': 'man', '高': 'gao', '低': 'di', '长': 'chang',
  '短': 'duan', '美': 'mei', '丑': 'chou', '胖': 'pang', '瘦': 'shou',
  '新': 'xin', '旧': 'jiu', '冷': 'leng', '热': 're', '难': 'nan',
  '易': 'yi', '开': 'kai', '关': 'guan', '进': 'jin', '出': 'chu',
  '吃': 'chi', '喝': 'he', '玩': 'wan', '乐': 'le', '睡': 'shui',
  '跑': 'pao', '走': 'zou', '飞': 'fei', '跳': 'tiao', '游': 'you',
  '神': 'shen', '鬼': 'gui', '仙': 'xian', '妖': 'yao', '魔': 'mo',
  '死': 'si', '活': 'huo', '生': 'sheng', '病': 'bing', '老': 'lao',
  '歌': 'ge', '舞': 'wu', '唱': 'chang', '弹': 'tan',
  '琴': 'qin', '棋': 'qi', '书': 'shu', '画': 'hua', '诗': 'shi',
  '弹幕': 'danmu', '主播': 'zhubo', '粉丝': 'fensi',
};

function toPinyin(text: string): string {
  let result = '';
  for (const ch of text) {
    if (PINYIN_MAP[ch]) {
      result += PINYIN_MAP[ch];
    } else {
      result += ch;
    }
  }
  return result;
}

// 变体字典 (谐音/缩写)
const VARIANTS: Record<string, string> = {
  'xswl': '笑死我了', 'yyds': '永远的神', 'awsl': '啊我死了',
  'u1s1': '有一说一', 'srds': '虽然但是', 'zqsg': '真情实感',
  'dbq': '对不起', 'nsdd': '你说得对', 'pljj': '漂亮姐姐',
  'nmd': '你妈的', 'wsnd': '我是你的',
  'tql': '太强了', 'sdl': '速度了',
};

function normalize(text: string): string {
  const lower = text.toLowerCase();
  if (VARIANTS[lower]) return VARIANTS[lower];
  return text;
}

export function basicCleanse(text: string, minLen = 1, maxLen = 128): string | null {
  if (!text) return null;

  // 移除控制字符 (保留换行/回车/制表)
  text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, '');
  // NFKC normalized (JS built-in)
  text = text.normalize('NFKC');
  // 折叠连续空白
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
  const cleaned = basicCleanse(rawText);
  if (!cleaned) return null;

  const variantNormalized = normalize(cleaned);
  const noCycle = compressCycle(variantNormalized);

  return { text: noCycle, normalized: noCycle };
}

/** SimHash: 64-bit fingerprint for near-duplicate detection */
export function computeSimhash(tokens: string[], bits = 64): bigint {
  const vec = new Int32Array(bits);
  for (const token of tokens) {
    let h = hashString(token);
    for (let i = 0; i < bits; i++) {
      if ((h >> BigInt(i)) & 1n) vec[i]++;
      else vec[i]--;
    }
  }
  let fingerprint = 0n;
  for (let i = 0; i < bits; i++) {
    if (vec[i] > 0) fingerprint |= (1n << BigInt(i));
  }
  return fingerprint;
}

export function hammingDistance(a: bigint, b: bigint): number {
  let xor = a ^ b;
  let dist = 0;
  while (xor > 0n) {
    dist++;
    xor &= xor - 1n;
  }
  return dist;
}

function hashString(s: string): bigint {
  // djb2 hash producing 64-bit value
  let h = 5381n;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5n) + h) ^ BigInt(s.charCodeAt(i));
  }
  return h & ((1n << 64n) - 1n);
}

/** DedupStore: canonical text → count */
export class DedupStore {
  private store = new Map<string, { count: number; raws: string[] }>();

  add(canonical: string, raw: string): boolean {
    const entry = this.store.get(canonical);
    if (entry) {
      entry.count++;
      if (!entry.raws.includes(raw)) entry.raws.push(raw);
      return false; // already exists
    }
    this.store.set(canonical, { count: 1, raws: [raw] });
    return true; // new
  }

  getCount(canonical: string): number {
    return this.store.get(canonical)?.count ?? 0;
  }

  get size(): number {
    return this.store.size;
  }

  clear() {
    this.store.clear();
  }
}
