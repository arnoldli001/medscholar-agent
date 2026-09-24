/* ============================================================================
 * MedScholar Agent — 前端应用逻辑（原生 ES2020，无任何依赖）
 * ----------------------------------------------------------------------------
 * 契约来源：docs/API.md（v1.0）。所有网络调用集中在「§3 API 层」，
 * 其余章节按界面区域划分，每节开头注明职责与所调用的端点。
 *
 * 设计原则：
 *   · 任何失败都以中文提示就地呈现，绝不让页面变空白；
 *   · 模型输出一律通过 DOM 节点构建（textContent），不使用 innerHTML 注入；
 *   · 所有副作用（定时器、EventSource、监听器）都有对应的清理路径。
 * ========================================================================== */
'use strict';

(function () {

/* ==========================================================================
 * §1. 常量与全局状态
 * ======================================================================== */

/** 数据源定义：键名与后端 medscholar/api/registry.py 的内部客户端名一致。 */
var SOURCE_DEFS = [
  { key: 'pubmed',           label: 'PubMed' },
  { key: 'europepmc',        label: 'Europe PMC' },
  { key: 'openalex',         label: 'OpenAlex' },
  { key: 'crossref',         label: 'Crossref' },
  { key: 'semantic_scholar', label: 'Semantic Scholar' },
  { key: 'arxiv',            label: 'arXiv' },
  { key: 'doaj',             label: 'DOAJ（开放获取期刊）' },
  { key: 'core',             label: 'CORE（机构库，需 key）' },
  { key: 'cnki',             label: 'CNKI（公开接口已失效）' }
];
var DEFAULT_SOURCES = ['pubmed', 'europepmc', 'openalex', 'crossref'];

/** 后端别名 → 内部规范名（models.SOURCE_LABELS 用 s2，registry 用 semantic_scholar）。 */
var SOURCE_ALIASES = { s2: 'semantic_scholar', semanticscholar: 'semantic_scholar', epmc: 'europepmc', pm: 'pubmed', cr: 'crossref', oa: 'openalex' };

/** 引用格式兜底表；启动后用 GET /api/cite/styles 覆盖。 */
var FALLBACK_STYLES = [
  { key: 'apa7', label: 'APA 7th' },
  { key: 'vancouver', label: 'Vancouver' },
  { key: 'gb7714', label: 'GB/T 7714-2015' },
  { key: 'chicago', label: 'Chicago (author-date)' },
  { key: 'bibtex', label: 'BibTeX' },
  { key: 'ris', label: 'RIS' }
];

/** 单次运行会收到的全部 SSE 事件类型（契约 §4）。 */
var SSE_TYPES = ['phase', 'status', 'token', 'plan', 'awaiting_approval', 'search_result',
                 'papers', 'critique', 'artifact', 'review', 'error', 'done'];

/** 工作流阶段中文名，用于阶段进度条。 */
var PHASE_LABELS = { plan: '规划', execute: '执行', reflect: '反思', synthesize: '综合', done: '完成' };
var PHASE_ORDER = ['plan', 'execute', 'reflect', 'synthesize', 'done'];

var STORE_KEYS = {
  topic: 'medscholar.topic',
  sources: 'medscholar.sources',
  style: 'medscholar.citeStyle',
  layout: 'medscholar.layout',
  settings: 'medscholar.settings',
  tab: 'medscholar.tab',
  searchMode: 'medscholar.searchMode',
  composerMode: 'medscholar.composerMode'
};

/** 应用状态（唯一可变数据源）。 */
var state = {
  settings: {
    apiBase: '', requireApproval: true, offline: false, debug: false,
    reviewMin: 4000, reviewMax: 8000,
    // 订阅资源只做"跳转打开"，不做下载：模板由用户填写
    libraryEnabled: false, libraryName: '图书馆', libraryOpenUrl: '', libraryProxyPrefix: ''
  },
  health: null,
  config: null,
  sources: DEFAULT_SOURCES.slice(),
  styles: FALLBACK_STYLES.slice(),
  citeStyle: 'gb7714',
  run: null,            // { id, sessionId, phase, finished, buffer, streamEl, startedAt }
  stream: null,         // EventSource
  reconnectTimer: null, // 断线后持续重连的定时器
  composerMode: 'ask',  // 'ask' = 知识库问答（默认），'research' = 完整研究流程
  approval: null,       // { runId, locked, el }
  toolEntries: 0,
  papers: [],           // 全部已收集文献（Paper 或 ScoredPaper 形态）
  paperIndex: {},       // key -> paper
  selected: {},         // key -> true
  citations: [],        // [{ index, text, paperKey }]
  artifacts: [],        // [{ id, title, fmt, content }]
  activeArtifact: 0,
  activeTab: 'papers',
  searchMode: 'live',
  sessions: [],
  activeSessionId: null,
  lastSearchOk: false,
  bannerDismissed: false
};

/* ==========================================================================
 * §2. 通用工具：DOM、格式化、提示
 * ======================================================================== */

var SVG_NS = 'http://www.w3.org/2000/svg';

/** 生成一个 `<svg><use href="#i-name"/></svg>` 图标节点。 */
function icon(name, cls) {
  var svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'ic ' + (cls || 'ic-sm'));
  svg.setAttribute('aria-hidden', 'true');
  var use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', '#i-' + name);
  svg.appendChild(use);
  return svg;
}

/** 创建元素：el('div', {class:'x'}, [子节点或字符串]) —— 文本一律用 textContent。 */
function el(tag, attrs, children) {
  var node = document.createElement(tag);
  if (attrs) {
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'style') node.setAttribute('style', v);
      else if (k === 'html') { /* 显式拒绝：避免误用 innerHTML */ node.textContent = v; }
      else if (k.indexOf('data-') === 0 || k === 'title' || k === 'type' || k === 'role' ||
               k === 'aria-label' || k === 'aria-selected' || k === 'aria-expanded' ||
               k === 'tabindex' || k === 'value' || k === 'placeholder' || k === 'colspan') {
        node.setAttribute(k, v);
      } else node[k] = v;
    });
  }
  appendChildren(node, children);
  return node;
}

function appendChildren(node, children) {
  if (children === null || children === undefined) return;
  if (!Array.isArray(children)) children = [children];
  children.forEach(function (c) {
    if (c === null || c === undefined || c === false) return;
    node.appendChild(typeof c === 'object' && c.nodeType ? c : document.createTextNode(String(c)));
  });
  return node;
}

/** 清空节点（跨浏览器安全，等价于 replaceChildren()）。 */
function clear(node) {
  if (!node) return node;
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function $(id) { return document.getElementById(id); }

/** 调试日志：仅在设置中开启时输出，避免刷屏。 */
function dbg() {
  if (!state.settings.debug) return;
  var args = Array.prototype.slice.call(arguments);
  args.unshift('[MedScholar]');
  try { console.log.apply(console, args); } catch (e) { /* 忽略 */ }
}

/** 两位补零。 */
function pad2(n) { return n < 10 ? '0' + n : String(n); }

/** 当前本地时间 HH:MM:SS。 */
function clockNow() {
  var d = new Date();
  return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
}

/** 把毫秒格式化为人类可读的耗时。 */
function fmtMs(ms) {
  var n = Number(ms);
  if (!isFinite(n) || n < 0) return '—';
  if (n < 1000) return Math.round(n) + ' ms';
  if (n < 60000) return (n / 1000).toFixed(n < 10000 ? 1 : 0) + ' s';
  return Math.floor(n / 60000) + ' 分 ' + Math.round((n % 60000) / 1000) + ' 秒';
}

/** 截断长文本。 */
function truncate(text, max) {
  var s = String(text === null || text === undefined ? '' : text);
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

/* --- 吐司提示 --- */
function toast(message, kind, timeout) {
  var stack = $('toastStack');
  if (!stack) return;
  var icons = { info: 'info', ok: 'check', warn: 'alert', error: 'alert' };
  var k = kind || 'info';
  var node = el('div', { class: 'toast toast-' + k, role: 'status' }, [
    icon(icons[k] || 'info'),
    el('div', { class: 'toast-body', text: message })
  ]);
  stack.appendChild(node);
  window.setTimeout(function () {
    node.classList.add('is-out');
    window.setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, 200);
  }, timeout || (k === 'error' ? 9000 : 4200));
}

/** 就地渲染一个可读的错误/信息块（用于面板内部，不打断整体布局）。 */
function noticeNode(kind, text, extraNodes) {
  var ic = kind === 'error' ? 'alert' : (kind === 'warn' ? 'alert' : 'info');
  var body = el('div', { class: 'notice-body' }, [el('span', { text: text })]);
  if (extraNodes && extraNodes.length) {
    body.appendChild(el('div', { class: 'notice-extra' }, extraNodes));
  }
  return el('div', { class: 'notice notice-' + kind }, [icon(ic), body]);
}

/** 在容器中显示一条错误提示（会清空容器）。 */
function showPanelError(container, text, retryFn, retryLabel) {
  if (!container) return;
  clear(container);
  var extras = [];
  if (retryFn) {
    extras.push(el('button', {
      class: 'btn btn-sm', type: 'button', text: retryLabel || '重试',
      onclick: retryFn
    }));
  }
  container.appendChild(noticeNode('error', text, extras));
}

/* --- localStorage 读写（隐私模式下可能抛异常，全部包住） --- */
function lsGet(key, fallback) {
  try {
    var raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw);
  } catch (e) { return fallback; }
}
function lsSet(key, value) {
  try { window.localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* 忽略配额/隐私模式错误 */ }
}

/** 读取并校验持久化设置。 */
function loadSettings() {
  var s = lsGet(STORE_KEYS.settings, null);
  if (s && typeof s === 'object') {
    state.settings.apiBase = typeof s.apiBase === 'string' ? s.apiBase : '';
    state.settings.requireApproval = s.requireApproval !== false;
    state.settings.offline = s.offline === true;
    state.settings.debug = s.debug === true;
    // 综述正文字数范围：默认 4000~8000 字（与后端 config 默认值一致）
    state.settings.reviewMin = toPositiveInt(s.reviewMin, 4000);
    state.settings.reviewMax = toPositiveInt(s.reviewMax, 8000);
    state.settings.libraryEnabled = s.libraryEnabled === true;
    state.settings.libraryName = typeof s.libraryName === 'string' ? s.libraryName : '图书馆';
    state.settings.libraryOpenUrl = typeof s.libraryOpenUrl === 'string' ? s.libraryOpenUrl : '';
    state.settings.libraryProxyPrefix = typeof s.libraryProxyPrefix === 'string' ? s.libraryProxyPrefix : '';
    if (state.settings.reviewMin > state.settings.reviewMax) {
      var swap = state.settings.reviewMin;
      state.settings.reviewMin = state.settings.reviewMax;
      state.settings.reviewMax = swap;
    }
  }
  var src = lsGet(STORE_KEYS.sources, null);
  if (Array.isArray(src) && src.length) {
    state.sources = src.map(normalizeSource).filter(function (v, i, a) { return a.indexOf(v) === i; });
  }
  var style = lsGet(STORE_KEYS.style, null);
  if (typeof style === 'string' && style) state.citeStyle = style;
  var tab = lsGet(STORE_KEYS.tab, null);
  if (typeof tab === 'string' && tab) state.activeTab = tab;
  var mode = lsGet(STORE_KEYS.searchMode, null);
  if (mode === 'live' || mode === 'local') state.searchMode = mode;
}
function saveSettings() { lsSet(STORE_KEYS.settings, state.settings); }

/** 把输入框里的值解析成正整数，非法时用兜底值。 */
function toPositiveInt(value, fallback) {
  var n = parseInt(String(value === undefined || value === null ? '' : value).trim(), 10);
  if (!isFinite(n) || n <= 0) return fallback;
  return n;
}

/** 把任意写法（s2 / 语义名）统一到规范数据源键。 */
function normalizeSource(name) {
  var key = String(name || '').trim().toLowerCase();
  return SOURCE_ALIASES[key] || key;
}

/** 数据源显示名。 */
function sourceLabel(key) {
  var k = normalizeSource(key);
  for (var i = 0; i < SOURCE_DEFS.length; i++) {
    if (SOURCE_DEFS[i].key === k) return SOURCE_DEFS[i].label;
  }
  return key || '未知来源';
}

/* ==========================================================================
 * §3. API 层
 * --------------------------------------------------------------------------
 * 所有请求都经由 apiFetch 统一处理：超时、JSON 解析、HTTP 错误、
 * 网络错误都转换成 { ok, data, error } 结构，调用方永远不会接到异常。
 * ======================================================================== */

/** 拼接接口地址：默认同源；设置里填了基地址则以其为准。 */
function apiUrl(path) {
  var base = (state.settings.apiBase || '').replace(/\/+$/, '');
  return base + path;
}

/**
 * 统一请求封装。
 * @returns {Promise<{ok:boolean,data:any,error:string,status:number}>}
 */
function apiFetch(path, options) {
  var opts = options || {};
  var method = opts.method || 'GET';
  var init = { method: method, headers: { 'Accept': 'application/json' } };
  if (opts.body !== undefined && opts.body !== null) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.body);
  }
  var timeoutMs = opts.timeout === undefined ? 45000 : opts.timeout;
  var controller = null;
  var timer = null;
  if (timeoutMs > 0 && typeof window.AbortController === 'function') {
    controller = new window.AbortController();
    init.signal = controller.signal;
    timer = window.setTimeout(function () { controller.abort(); }, timeoutMs);
  }
  var url = apiUrl(path);
  dbg(method, url, opts.body || '');

  return window.fetch(url, init).then(function (res) {
    return res.text().then(function (text) {
      var data = null;
      if (text) { try { data = JSON.parse(text); } catch (e) { data = null; } }
      if (!res.ok) {
        var detail = data && (data.detail || data.message || data.error);
        return {
          ok: false, status: res.status, data: data,
          error: detail ? String(detail) : ('请求失败：HTTP ' + res.status + ' ' + (res.statusText || ''))
        };
      }
      if (data === null && text) {
        return { ok: false, status: res.status, data: null, error: '响应不是合法 JSON，后端可能未正常运行。' };
      }
      return { ok: true, status: res.status, data: data, error: '' };
    });
  }).catch(function (err) {
    var msg;
    if (err && err.name === 'AbortError') {
      msg = '请求超时（' + Math.round(timeoutMs / 1000) + ' 秒未响应）。若正在检索多个外部数据源，可稍后重试或减少数据源。';
    } else {
      msg = '无法连接后端服务（' + (err && err.message ? err.message : '网络错误') + '）。' +
            '请确认 MedScholar 服务已在本机运行，且接口地址为 ' + (state.settings.apiBase || '当前页面同源') + '。';
    }
    return { ok: false, status: 0, data: null, error: msg };
  }).then(function (result) {
    if (timer) window.clearTimeout(timer);
    return result;
  });
}

/* --- 具体端点（每个函数对应契约中的一条路径） --- */

var API = {
  /** GET /api/ping —— 极轻量的连通性探测（不读数据库、不碰模型） */
  ping: function () { return apiFetch('/api/ping', { timeout: 5000 }); },
  /** POST /api/query/preview —— 解析检索式并回显各库语法 */
  queryPreview: function (query, sources) {
    return apiFetch('/api/query/preview', {
      method: 'POST',
      body: { query: query, sources: sources || [] },
      timeout: 8000
    });
  },
  /** GET /api/health —— 健康状态、知识库统计、数据源清单 */
  health: function (opts) {
    var probe = opts && opts.probe ? '?probe=true' : '';
    return apiFetch('/api/health' + probe, { timeout: 20000 });
  },
  /** GET /api/config —— 脱敏配置 */
  config: function () { return apiFetch('/api/config', { timeout: 15000 }); },
  /** GET /api/stats —— 数据库统计 */
  stats: function () { return apiFetch('/api/stats', { timeout: 15000 }); },
  /** GET /api/metrics —— 运行指标（LLM 用量与成本 / 熔断 / 缓存命中率 / 注入扫描） */
  metrics: function () { return apiFetch('/api/metrics', { timeout: 15000 }); },
  /** GET /api/cite/styles —— 可用引用格式 */
  citeStyles: function () { return apiFetch('/api/cite/styles', { timeout: 15000 }); },
  /** GET /api/papers —— 文献库分页/排序/过滤 */
  papers: function (params) { return apiFetch('/api/papers?' + toQuery(params)); },
  /** POST /api/papers/search —— 本地知识库混合检索（BM25 + 向量 + RRF） */
  searchLocal: function (body) { return apiFetch('/api/papers/search', { method: 'POST', body: body, timeout: 60000 }); },
  /** GET /api/papers/{id} —— 单篇详情 */
  paper: function (id) { return apiFetch('/api/papers/' + encodeURIComponent(id)); },
  /** POST /api/search/live —— 联网多源检索 */
  searchLive: function (body) { return apiFetch('/api/search/live', { method: 'POST', body: body, timeout: 300000 }); },
  /** POST /api/agent/run —— 启动研究工作流 */
  agentRun: function (body) { return apiFetch('/api/agent/run', { method: 'POST', body: body, timeout: 30000 }); },
  /** GET /api/agent/runs —— 实时运行 + 历史记录（含被服务重启中断的） */
  agentRuns: function () { return apiFetch('/api/agent/runs', { timeout: 15000 }); },
  /** POST /api/agent/approve/{run_id} —— 审批（approve / revise / cancel） */
  agentApprove: function (runId, body) {
    return apiFetch('/api/agent/approve/' + encodeURIComponent(runId), { method: 'POST', body: body, timeout: 30000 });
  },
  /** POST /api/agent/cancel/{run_id} —— 中止运行 */
  agentCancel: function (runId) {
    return apiFetch('/api/agent/cancel/' + encodeURIComponent(runId), { method: 'POST', timeout: 20000 });
  },
  /** GET /api/sessions —— 会话列表 */
  sessions: function () { return apiFetch('/api/sessions'); },
  /** GET /api/sessions/{id}/messages —— 会话消息 */
  sessionMessages: function (id) { return apiFetch('/api/sessions/' + encodeURIComponent(id) + '/messages'); },
  /** DELETE /api/sessions/{id} */
  deleteSession: function (id) { return apiFetch('/api/sessions/' + encodeURIComponent(id), { method: 'DELETE' }); },
  /** GET /api/artifacts?session_id= —— 产物列表 */
  artifacts: function (sessionId) {
    return apiFetch('/api/artifacts' + (sessionId ? '?session_id=' + encodeURIComponent(sessionId) : ''));
  },
  /** GET /api/artifacts/{id} —— 产物正文 */
  artifact: function (id) { return apiFetch('/api/artifacts/' + encodeURIComponent(id)); },
  /** GET /api/agent/latest —— 最近一次运行及其阶段快照摘要（页面加载时恢复现场） */
  agentLatest: function () { return apiFetch('/api/agent/latest'); },
  /** GET /api/agent/steps/{run_id} —— 各阶段快照摘要 */
  agentSteps: function (runId) { return apiFetch('/api/agent/steps/' + encodeURIComponent(runId)); },
  /** POST /api/agent/resume/{run_id} —— 从中断处继续（不重跑已完成阶段） */
  agentResume: function (runId) {
    return apiFetch('/api/agent/resume/' + encodeURIComponent(runId), { method: 'POST', timeout: 60000 });
  },
  /** POST /api/import —— 导入题录文件 */
  importPapers: function (body) {
    // 大文件 + 逐条嵌入向量，给足超时
    return apiFetch('/api/import', { method: 'POST', body: body, timeout: 900000 });
  },
  /** POST /api/feedback —— 赞/踩/质疑 */
  feedback: function (body) {
    return apiFetch('/api/feedback', { method: 'POST', body: body, timeout: 30000 });
  },
  /** GET /api/feedback/summary —— 学习闭环进度 */
  feedbackSummary: function () { return apiFetch('/api/feedback/summary'); },
  /** GET /api/feedback/export —— 导出偏好对 */
  feedbackExport: function () { return apiFetch('/api/feedback/export?limit=1000'); },
  /** POST /api/manuscript/draft —— 按实验数据生成论文初稿 */
  manuscriptDraft: function (body) {
    return apiFetch('/api/manuscript/draft', { method: 'POST', body: body, timeout: 1800000 });
  },
  /** POST /api/cite —— 生成引用 */
  cite: function (body) { return apiFetch('/api/cite', { method: 'POST', body: body, timeout: 60000 }); },
  /** POST /api/export —— 导出文件 */
  exportPapers: function (body) { return apiFetch('/api/export', { method: 'POST', body: body, timeout: 120000 }); },
  /** POST /api/maintenance/embed —— 补齐向量 */
  embed: function (body) { return apiFetch('/api/maintenance/embed', { method: 'POST', body: body, timeout: 600000 }); },
  /** POST /api/maintenance/fulltext —— 为开放获取文献补齐全文 */
  fulltextBackfill: function (body) {
    return apiFetch('/api/maintenance/fulltext', { method: 'POST', body: body, timeout: 1800000 });
  },
  /** POST /api/maintenance/reindex —— 重建索引 */
  reindex: function () { return apiFetch('/api/maintenance/reindex', { method: 'POST', timeout: 600000 }); }
};

/** 对象 → 查询串（跳过空值）。 */
function toQuery(params) {
  var parts = [];
  Object.keys(params || {}).forEach(function (k) {
    var v = params[k];
    if (v === null || v === undefined || v === '') return;
    parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(v));
  });
  return parts.join('&');
}

/* ==========================================================================
 * §4. Markdown 渲染器
 * --------------------------------------------------------------------------
 * 手写、安全：所有文本都经 document.createTextNode 写入，从不使用 innerHTML。
 * 支持：# / ## / ### 标题、**粗体**、*斜体*、`行内代码`、- 无序列表、
 * 1. 有序列表、> 引用、--- 分隔线、| 表格 |，以及 [1] / [1,2] / [1-3] 引用标记。
 * ======================================================================== */

var MD_INLINE_RE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\n]+\*|\[\d+(?:\s*[,\-–]\s*\d+)*\])/g;

/** 把一段行内文本解析为 DOM 节点数组。 */
function appendInline(container, text) {
  var src = String(text === null || text === undefined ? '' : text);
  var last = 0;
  var m;
  MD_INLINE_RE.lastIndex = 0;
  while ((m = MD_INLINE_RE.exec(src)) !== null) {
    if (m.index > last) container.appendChild(document.createTextNode(src.slice(last, m.index)));
    var tok = m[0];
    if (tok.indexOf('**') === 0) {
      container.appendChild(el('strong', { text: tok.slice(2, -2) }));
    } else if (tok.charAt(0) === '`') {
      container.appendChild(el('code', { class: 'md-code', text: tok.slice(1, -1) }));
    } else if (tok.charAt(0) === '*') {
      container.appendChild(el('em', { text: tok.slice(1, -1) }));
    } else {
      container.appendChild(buildCiteMarker(tok));
    }
    last = m.index + tok.length;
  }
  if (last < src.length) container.appendChild(document.createTextNode(src.slice(last)));
  return container;
}

/** 把 `[1,2]` / `[1-3]` 渲染成可点击的引用标记。 */
function buildCiteMarker(token) {
  var inner = token.slice(1, -1);
  var nums = [];
  inner.split(/\s*,\s*/).forEach(function (part) {
    var p = part.trim();
    var range = /^(\d+)\s*[-–]\s*(\d+)$/.exec(p);
    if (range) {
      var a = parseInt(range[1], 10);
      var b = parseInt(range[2], 10);
      if (b >= a && b - a <= 60) {
        for (var i = a; i <= b; i++) nums.push(i);
        return;
      }
    }
    if (/^\d+$/.test(p)) nums.push(parseInt(p, 10));
  });
  if (!nums.length) return document.createTextNode(token);

  var span = el('span', { class: 'cite-mark' }, ['[']);
  nums.forEach(function (n, idx) {
    if (idx) span.appendChild(el('span', { class: 'cite-mark-sep', text: ',' }));
    span.appendChild(el('button', {
      type: 'button',
      class: 'cite-mark-btn',
      'data-cite-index': String(n),
      title: '跳转到引用列表第 ' + n + ' 条',
      text: String(n)
    }));
  });
  span.appendChild(document.createTextNode(']'));
  return span;
}

/** 渲染 Markdown 文本，返回 DocumentFragment。 */
function renderMarkdown(text) {
  var frag = document.createDocumentFragment();
  var lines = String(text === null || text === undefined ? '' : text).replace(/\r\n?/g, '\n').split('\n');
  var listEl = null;
  var listTag = '';

  function closeList() { listEl = null; listTag = ''; }

  for (var i = 0; i < lines.length; i++) {
    var raw = lines[i];
    var line = raw.replace(/\s+$/, '');
    var m;

    if (!line.trim()) { closeList(); continue; }

    // 标题
    m = /^(#{1,4})\s+(.*)$/.exec(line);
    if (m) {
      closeList();
      var level = m[1].length;
      var h = el('h' + Math.min(level + 1, 6), { class: 'md-h md-h' + level });
      appendInline(h, m[2]);
      frag.appendChild(h);
      continue;
    }

    // 分隔线
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
      closeList();
      frag.appendChild(el('hr', { class: 'md-hr' }));
      continue;
    }

    // 引用块
    m = /^>\s?(.*)$/.exec(line);
    if (m) {
      closeList();
      var bq = el('blockquote', { class: 'md-blockquote' });
      appendInline(bq, m[1]);
      frag.appendChild(bq);
      continue;
    }

    // 表格：连续的以 | 开头的行视为一张表
    if (line.trim().charAt(0) === '|') {
      closeList();
      var tableLines = [line];
      while (i + 1 < lines.length && lines[i + 1].trim().charAt(0) === '|') {
        i++;
        tableLines.push(lines[i]);
      }
      frag.appendChild(buildTable(tableLines));
      continue;
    }

    // 无序列表
    m = /^\s*[-*+]\s+(.*)$/.exec(line);
    if (m) {
      if (listTag !== 'ul') { closeList(); listEl = el('ul', { class: 'md-ul' }); listTag = 'ul'; frag.appendChild(listEl); }
      var liU = el('li');
      appendInline(liU, m[1]);
      listEl.appendChild(liU);
      continue;
    }

    // 有序列表
    m = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (m) {
      if (listTag !== 'ol') { closeList(); listEl = el('ol', { class: 'md-ol' }); listTag = 'ol'; frag.appendChild(listEl); }
      var liO = el('li');
      appendInline(liO, m[1]);
      listEl.appendChild(liO);
      continue;
    }

    // 普通段落
    closeList();
    var p = el('p', { class: 'md-p' });
    appendInline(p, line);
    frag.appendChild(p);
  }
  return frag;
}

/** 由若干 `| a | b |` 行构建表格（首行为表头，第二行若为分隔行则跳过）。 */
function buildTable(tableLines) {
  function cells(line) {
    var t = line.trim();
    if (t.charAt(0) === '|') t = t.slice(1);
    if (t.charAt(t.length - 1) === '|') t = t.slice(0, -1);
    return t.split('|').map(function (c) { return c.trim(); });
  }
  var header = cells(tableLines[0]);
  var bodyStart = 1;
  if (tableLines.length > 1 && /^[\s|:\-]+$/.test(tableLines[1])) bodyStart = 2;

  var table = el('table', { class: 'md-table' });
  var thead = el('thead');
  var htr = el('tr');
  header.forEach(function (c) {
    var th = el('th');
    appendInline(th, c);
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  table.appendChild(thead);

  var tbody = el('tbody');
  for (var i = bodyStart; i < tableLines.length; i++) {
    var tr = el('tr');
    cells(tableLines[i]).forEach(function (c) {
      var td = el('td');
      appendInline(td, c);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return table;
}

/** 清空容器并渲染 Markdown。 */
function setMarkdown(container, text) {
  clear(container);
  container.appendChild(renderMarkdown(text));
  return container;
}

/* ==========================================================================
 * §5. 布局：栏宽拖拽与持久化
 * --------------------------------------------------------------------------
 * 两条 gutter 使用 Pointer Events 拖拽，宽度写入 CSS 变量并持久化到
 * localStorage；同时支持键盘方向键微调（可访问性）。
 * ======================================================================== */

var LAYOUT_LIMITS = { leftMin: 190, leftMax: 460, midMin: 320, rightMin: 280 };

function loadLayout() {
  var saved = lsGet(STORE_KEYS.layout, null);
  var root = document.documentElement;
  if (saved && typeof saved === 'object') {
    if (typeof saved.left === 'number') root.style.setProperty('--left-w', clamp(saved.left, LAYOUT_LIMITS.leftMin, LAYOUT_LIMITS.leftMax) + 'px');
    if (typeof saved.mid === 'number') root.style.setProperty('--mid-w', saved.mid + 'px');
  }
}
function saveLayout() {
  var root = document.documentElement;
  var left = parseInt(root.style.getPropertyValue('--left-w'), 10) || 268;
  var mid = parseInt(root.style.getPropertyValue('--mid-w'), 10) || 560;
  lsSet(STORE_KEYS.layout, { left: left, mid: mid });
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** 绑定一个 gutter 的拖拽行为。which = 'left' | 'right'。 */
function bindGutter(gutter, which) {
  if (!gutter) return;
  var dragging = false;
  var startX = 0;
  var startLeft = 0;
  var startMid = 0;
  var root = document.documentElement;

  function currentLeft() { return parseInt(root.style.getPropertyValue('--left-w'), 10) || gutterWidth('left'); }
  function currentMid() { return parseInt(root.style.getPropertyValue('--mid-w'), 10) || gutterWidth('mid'); }
  function gutterWidth(kind) {
    var v = window.getComputedStyle(root).getPropertyValue(kind === 'left' ? '--left-w' : '--mid-w');
    return parseInt(v, 10) || (kind === 'left' ? 268 : 560);
  }

  function onMove(ev) {
    if (!dragging) return;
    var dx = ev.clientX - startX;
    if (which === 'left') {
      root.style.setProperty('--left-w', clamp(startLeft + dx, LAYOUT_LIMITS.leftMin, LAYOUT_LIMITS.leftMax) + 'px');
    } else {
      // 中栏右边界拖动：中栏变宽时右侧自然收窄，但给右栏留出最小宽度
      var workbench = $('workbench');
      var total = workbench ? workbench.clientWidth : window.innerWidth;
      var maxMid = Math.max(LAYOUT_LIMITS.midMin, total - currentLeft() - LAYOUT_LIMITS.rightMin - 20);
      root.style.setProperty('--mid-w', clamp(startMid + dx, LAYOUT_LIMITS.midMin, maxMid) + 'px');
    }
    ev.preventDefault();
  }
  function onUp() {
    if (!dragging) return;
    dragging = false;
    gutter.classList.remove('is-dragging');
    document.body.classList.remove('is-resizing');
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onUp);
    saveLayout();
  }
  function onDown(ev) {
    if (ev.button !== 0) return;
    dragging = true;
    startX = ev.clientX;
    startLeft = currentLeft();
    startMid = currentMid();
    gutter.classList.add('is-dragging');
    document.body.classList.add('is-resizing');
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    ev.preventDefault();
  }

  gutter.addEventListener('pointerdown', onDown);
  gutter.addEventListener('dblclick', function () {
    root.style.setProperty('--left-w', '268px');
    root.style.setProperty('--mid-w', '560px');
    saveLayout();
  });
  // 键盘可访问性：方向键调整栏宽
  gutter.addEventListener('keydown', function (ev) {
    var step = ev.shiftKey ? 32 : 12;
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    var dir = ev.key === 'ArrowRight' ? 1 : -1;
    if (which === 'left') {
      root.style.setProperty('--left-w', clamp(currentLeft() + dir * step, LAYOUT_LIMITS.leftMin, LAYOUT_LIMITS.leftMax) + 'px');
    } else {
      root.style.setProperty('--mid-w', Math.max(LAYOUT_LIMITS.midMin, currentMid() + dir * step) + 'px');
    }
    saveLayout();
    ev.preventDefault();
  });
}

/* ==========================================================================
 * §6. 健康状态与配置
 * --------------------------------------------------------------------------
 * 端点：GET /api/health、GET /api/config、GET /api/stats、GET /api/cite/styles
 * 职责：健康横幅、知识库统计卡片、底栏元数据、数据源下拉的启用状态。
 * ======================================================================== */

function refreshHealth(options) {
  var silent = options && options.silent;
  return API.health().then(function (res) {
    if (!res.ok) {
      state.health = null;
      setHealthDot('bad');
      renderKbStats(null, res.error);
      renderBanner([{
        text: '无法获取后端健康状态：' + res.error,
        hint: '请确认服务已启动（默认 http://127.0.0.1:8760），或在「设置」中修改接口基地址。'
      }]);
      renderMetaStrip();
      if (!silent) toast('后端不可用：' + truncate(res.error, 120), 'error');
      return null;
    }
    var h = res.data || {};
    state.health = h;
    var verdict = healthVerdict(h);
    setHealthDot(verdict.state);
    renderKbStats(h, null);
    renderBanner(verdict.problems);
    renderMetaStrip();
    renderHealthDetail(h);
    renderSourceMenu();
    renderSourceNote();
    if (h.version) {
      var v = $('brandVersion');
      if (v) { v.textContent = 'v' + h.version; v.hidden = false; }
      var sv = $('settingsVersion');
      if (sv) sv.textContent = 'MedScholar Agent v' + h.version + (h.offline ? ' · 离线模式' : '');
    }
    var note = $('welcomeHealthNote');
    if (note) {
      note.textContent = verdict.problems.length
        ? '当前运行状态有降级项，详见顶部提示条。'
        : '后端与模型均已就绪，可以直接开始检索。';
    }
    if (!silent) dbg('health', h);
    return h;
  });
}

/** 依据 /api/health 判定总体状态并产出人类可读的问题清单。 */
function healthVerdict(h) {
  var problems = [];
  var level = 'ok';
  if (!h) return { state: 'bad', problems: [{ text: '未取得健康状态。' }] };

  if (h.offline) {
    problems.push({
      text: '服务运行在离线模式：不会访问 PubMed 等外部数据库，检索仅覆盖本地知识库。',
      hint: '如需联网检索，请在「设置 → 运行设置」中关闭离线模式后重启服务。'
    });
    level = 'warn';
  }
  if (h.llm && h.llm.ok === false) {
    problems.push({
      text: '大语言模型不可用（' + (h.llm.provider || '未知提供方') + (h.llm.model ? ' / ' + h.llm.model : '') + '）：' + (h.llm.message || '未提供原因'),
      hint: '规划、论证与综述撰写将无法进行。若使用 Ollama，请先启动服务并拉取模型（例如 ollama pull qwen3:8b）。'
    });
    level = 'warn';
  } else if (h.llm && h.llm.ok === null) {
    // 后端改为后台异步探测模型，首屏会短暂返回 null；这不是故障，不要吓唬用户
    problems.push({
      text: '正在检测大语言模型（' + (h.llm.provider || '') + (h.llm.model ? ' / ' + h.llm.model : '') + '）…',
      hint: '首次使用或模型刚被卸载时需要重新载入内存，通常几秒到半分钟。'
    });
    if (level === 'ok') level = 'warn';
  }
  if (h.embedding && h.embedding.ok === false) {
    problems.push({
      text: '向量模型不可用（' + (h.embedding.provider || '未知提供方') + (h.embedding.model ? ' / ' + h.embedding.model : '') + '）：' + (h.embedding.message || '未提供原因'),
      hint: '本地知识库将退化为仅 BM25 关键词检索，语义召回会明显下降。'
    });
    level = 'warn';
  }

  var db = h.database || {};
  var papers = Number(db.papers || 0);
  var embedded = Number(db.embedded || 0);
  if (papers === 0) {
    problems.push({
      text: '知识库中还没有文献。',
      hint: '先在上方「联网检索」中执行一次检索并勾选「入库」，或直接启动研究工作流。'
    });
    if (level === 'ok') level = 'warn';
  } else if (embedded < papers) {
    problems.push({
      text: '知识库中有 ' + (papers - embedded) + ' / ' + papers + ' 篇文献尚未生成向量，向量检索覆盖不完整。',
      hint: '可在「设置 → 知识库维护」中点击「补齐向量」。'
    });
    if (level === 'ok') level = 'warn';
  }

  var disabled = (h.sources || []).filter(function (s) { return s && s.enabled === false; });
  if (disabled.length) {
    problems.push({
      text: '以下数据源已禁用，检索时会被跳过：' + disabled.map(function (s) { return s.label || s.name; }).join('、') + '。',
      hint: '可在服务端配置中启用，或在前端取消勾选以避免混淆。'
    });
    if (level === 'ok') level = 'warn';
  }
  return { state: level, problems: problems };
}

function setHealthDot(s) {
  var dot = $('healthDot');
  if (dot) dot.setAttribute('data-state', s);
}

/** 渲染顶部琥珀色健康横幅；无问题时隐藏。 */
function renderBanner(problems) {
  var banner = $('healthBanner');
  if (!banner) return;
  if (!problems || !problems.length || state.bannerDismissed) { banner.hidden = true; return; }
  banner.hidden = false;
  var list = $('healthBannerList');
  clear(list);
  problems.forEach(function (p) {
    var li = el('li', null, [
      el('span', null, [
        el('span', { text: p.text }),
        p.hint ? el('span', { class: 'banner-hint', text: ' ' + p.hint }) : null
      ])
    ]);
    list.appendChild(li);
  });
  var title = $('healthBannerTitle');
  if (title) {
    title.textContent = problems.length > 1
      ? '运行状态提示（' + problems.length + ' 项降级）'
      : '运行状态提示';
  }
}

/** 左栏底部「知识库」统计卡片。 */
function renderKbStats(h, errorText) {
  var box = $('kbStats');
  var backend = $('kbBackend');
  if (!box) return;
  clear(box);
  if (!h) {
    if (backend) backend.textContent = '离线';
    box.appendChild(noticeNode('error', errorText || '未取得知识库统计。'));
    return;
  }
  var db = h.database || {};
  var papers = Number(db.papers || 0);
  var embedded = Number(db.embedded || 0);
  var pct = papers > 0 ? Math.round(embedded / papers * 100) : 0;
  if (backend) backend.textContent = db.vector_backend || '未知后端';

  var grid = el('div', { class: 'kb-grid' }, [
    el('div', { class: 'kb-cell' }, [
      el('span', { class: 'kb-key', text: '文献总数' }),
      el('span', { class: 'kb-val', text: String(papers) })
    ]),
    el('div', { class: 'kb-cell' }, [
      el('span', { class: 'kb-key', text: '已生成向量' }),
      el('span', { class: 'kb-val', text: String(embedded) })
    ]),
    el('div', { class: 'kb-cell' }, [
      el('span', { class: 'kb-key', text: '数据库体积' }),
      el('span', { class: 'kb-val', text: (db.db_size_mb === undefined || db.db_size_mb === null ? '—' : db.db_size_mb + ' MB') })
    ]),
    el('div', { class: 'kb-cell' }, [
      el('span', { class: 'kb-key', text: '向量后端' }),
      el('span', { class: 'kb-val kb-val-sm', text: db.vector_backend || '—' })
    ])
  ]);
  box.appendChild(grid);

  var cov = el('div', { class: 'coverage' }, [
    el('div', { class: 'coverage-head' }, [
      el('span', { text: '向量覆盖率' }),
      el('span', { text: papers ? pct + '%' : '—' })
    ]),
    el('div', { class: 'coverage-bar' }, [
      el('div', { class: 'coverage-fill', style: 'width:' + (papers ? pct : 0) + '%' })
    ])
  ]);
  box.appendChild(cov);

  if (db.db_path) {
    box.appendChild(el('p', { class: 'kb-path', title: db.db_path, text: db.db_path }));
  }
}

/**
 * 单独刷新知识库统计：GET /api/stats。
 * 契约 §8 说明该端点返回「同 /api/health 的 database 字段」，但未明确是否外层再包一层
 * database 键，因此这里对两种形态都做兼容。
 */
function refreshKbStats() {
  var btn = $('refreshKbBtn');
  if (btn) btn.disabled = true;
  return API.stats().then(function (res) {
    if (btn) btn.disabled = false;
    if (!res.ok) {
      toast('知识库统计刷新失败：' + truncate(res.error, 100), 'error');
      return;
    }
    var payload = res.data || {};
    var db = payload.database && typeof payload.database === 'object' ? payload.database : payload;
    renderKbStats({ database: db }, null);
    renderMetaStrip();
    toast('知识库统计已刷新。', 'ok');
  });
}

function renderMetaStrip() {
  var h = state.health;
  var m = $('metaModel'), e = $('metaEmbed'), p = $('metaPapers');
  if (!h) {
    if (m) m.textContent = '模型：不可用';
    if (e) e.textContent = '向量：不可用';
    if (p) p.textContent = '文献：—';
    return;
  }
  if (m) m.textContent = '模型：' + ((h.llm && h.llm.ok) ? (h.llm.model || h.llm.provider || '—') : '不可用');
  if (e) e.textContent = '向量：' + ((h.embedding && h.embedding.ok) ? (h.embedding.model || '—') + (h.embedding.dim ? ' (' + h.embedding.dim + 'd)' : '') : '不可用');
  var db = h.database || {};
  if (p) p.textContent = '文献：' + (db.papers === undefined ? '—' : db.papers + ' 篇 / 向量 ' + (db.embedded === undefined ? '—' : db.embedded));
}

/** 设置对话框「健康状态」页详情。 */
function renderHealthDetail(h) {
  var box = $('healthDetail');
  if (!box) return;
  clear(box);
  if (!h) { box.appendChild(noticeNode('error', '未取得健康状态。')); return; }

  function row(name, ok, msg, detail) {
    var cls = ok === true ? 'is-ok' : (ok === false ? 'is-bad' : 'is-warn');
    return el('div', { class: 'health-row ' + cls }, [
      icon(ok === true ? 'check' : (ok === false ? 'x' : 'alert')),
      el('div', { class: 'health-row-main' }, [
        el('div', { class: 'health-row-name', text: name + (detail ? ' · ' + detail : '') }),
        el('div', { class: 'health-row-msg', text: msg || '' })
      ])
    ]);
  }

  var grid = el('div', { class: 'health-grid' });
  grid.appendChild(row('服务', true, '版本 ' + (h.version || '—') + (h.offline ? ' · 当前离线模式' : ' · 允许联网检索')));
  grid.appendChild(row('大语言模型', !!(h.llm && h.llm.ok),
    (h.llm && h.llm.message) || '—',
    h.llm ? ((h.llm.provider || '') + (h.llm.model ? ' / ' + h.llm.model : '')) : ''));
  grid.appendChild(row('向量模型', !!(h.embedding && h.embedding.ok),
    (h.embedding && h.embedding.message) || '—',
    h.embedding ? ((h.embedding.provider || '') + (h.embedding.model ? ' / ' + h.embedding.model : '') + (h.embedding.dim ? ' · ' + h.embedding.dim + 'd' : '')) : ''));

  var db = h.database || {};
  var dbRow = row('知识库', Number(db.papers || 0) > 0,
    '文献 ' + (db.papers === undefined ? '—' : db.papers) + ' 篇 · 已向量化 ' + (db.embedded === undefined ? '—' : db.embedded) +
    ' · 体积 ' + (db.db_size_mb === undefined ? '—' : db.db_size_mb + ' MB') +
    (db.db_path ? '\n' + db.db_path : ''),
    db.vector_backend || '');
  grid.appendChild(dbRow);

  var srcs = h.sources || [];
  if (srcs.length) {
    var tbl = el('div', { class: 'health-src-table' });
    tbl.appendChild(el('div', { class: 'form-label', text: '数据源' }));
    srcs.forEach(function (s) {
      tbl.appendChild(el('div', { class: 'health-src' }, [
        el('span', { text: (s.label || s.name || '未知') + (/cnki/i.test(s.name || '') ? '（可能受网络限制）' : '') }),
        el('span', { text: s.has_api_key ? '已配置密钥' : '免密钥' }),
        el('span', { class: s.enabled === false ? 'chip chip-bad' : 'chip chip-ok', text: s.enabled === false ? '已禁用' : '启用 · ' + (s.rps ? s.rps + ' rps' : '') })
      ]));
    });
    grid.appendChild(tbl);
  }

  if (h.sources === undefined && h.database === undefined) {
    grid.appendChild(noticeNode('warn', '健康响应缺少 database / sources 字段，可能是后端版本较旧。'));
  }
  box.appendChild(grid);
}

/** 拉取引用格式清单。 */
function refreshCiteStyles() {
  return API.citeStyles().then(function (res) {
    if (!res.ok || !res.data || !Array.isArray(res.data.styles) || !res.data.styles.length) {
      dbg('cite styles fallback', res.error);
      return null;
    }
    state.styles = res.data.styles.filter(function (s) { return s && s.key; });
    renderStyleSelect();
    return state.styles;
  });
}

function renderStyleSelect() {
  var sel = $('citeStyleSelect');
  if (!sel) return;
  var styles = state.styles.length ? state.styles : FALLBACK_STYLES;
  clear(sel);
  styles.forEach(function (s) {
    sel.appendChild(el('option', { value: s.key, text: s.label || s.key }));
  });
  var known = styles.some(function (s) { return s.key === state.citeStyle; });
  if (!known) state.citeStyle = styles[0].key;
  sel.value = state.citeStyle;
  updateCiteStyleLabel();
}

function updateCiteStyleLabel() {
  var label = state.citeStyle;
  state.styles.forEach(function (s) { if (s.key === state.citeStyle) label = s.label || s.key; });
  var node = $('citeStyleName');
  if (node) node.textContent = label;
  return label;
}

/* ==========================================================================
 * §7. 左栏
 * --------------------------------------------------------------------------
 * 会话历史：GET /api/sessions、GET /api/sessions/{id}/messages、DELETE /api/sessions/{id}
 * 文献库：  GET /api/papers
 * 知识库：  GET /api/health（见 §6）
 * ======================================================================== */

/** 渲染会话列表。 */
function refreshSessions() {
  var box = $('sessionList');
  if (box) showPanelError(box, '正在加载会话…');
  if (box) clear(box);
  return API.sessions().then(function (res) {
    if (!box) return;
    clear(box);
    if (!res.ok) {
      showPanelError(box, '会话列表加载失败：' + res.error, refreshSessions, '重试');
      return;
    }
    var items = (res.data && res.data.items) || [];
    state.sessions = items;
    if (!items.length) {
      box.appendChild(el('p', { class: 'empty-hint', text: '还没有会话记录。启动一次研究工作流后会自动创建。' }));
      return;
    }
    items.forEach(function (s) { box.appendChild(sessionRow(s)); });
  });
}

function sessionRow(s) {
  var del = el('button', {
    type: 'button', class: 'btn btn-icon btn-sm session-del', title: '删除该会话', 'aria-label': '删除会话',
    onclick: function (ev) {
      ev.stopPropagation();
      deleteSession(s.id);
    }
  }, [icon('trash', 'ic-xs')]);

  return el('div', {
    class: 'session-item' + (state.activeSessionId === s.id ? ' is-active' : ''),
    'data-session-id': s.id,
    title: s.topic || s.title || '',
    onclick: function () { openSession(s.id); }
  }, [
    el('div', { class: 'session-main' }, [
      el('span', { class: 'session-title', text: s.title || s.topic || ('会话 #' + s.id) }),
      el('span', { class: 'session-meta' }, [
        el('span', { text: s.message_count === undefined ? '' : s.message_count + ' 条消息' }),
        el('span', { text: s.updated_at || s.created_at || '' })
      ])
    ]),
    del
  ]);
}

/** 打开会话：读取消息并恢复文献/产物。 */
function openSession(id) {
  if (!id) return;
  state.activeSessionId = id;
  var box = $('sessionList');
  if (box) {
    Array.prototype.forEach.call(box.querySelectorAll('.session-item'), function (n) {
      n.classList.toggle('is-active', String(n.getAttribute('data-session-id')) === String(id));
    });
  }
  pushStatus('正在载入会话 #' + id + ' 的消息…');
  API.sessionMessages(id).then(function (res) {
    if (!res.ok) {
      pushError('会话消息加载失败：' + res.error);
      return;
    }
    var items = (res.data && res.data.items) || [];
    if (!items.length) {
      pushStatus('该会话暂无消息。');
    } else {
      items.forEach(function (m) {
        var role = m.role === 'user' ? 'user' : (m.role === 'assistant' ? 'agent' : 'system');
        pushMessage(role, m.content, m.created_at);
      });
    }
  });
  API.artifacts(id).then(function (res) {
    if (!res.ok || !res.data || !Array.isArray(res.data.items)) return;
    res.data.items.forEach(function (a) {
      if (!a || a.id === undefined) return;
      var known = state.artifacts.some(function (x) { return x.id === a.id; });
      if (!known) state.artifacts.push({ id: a.id, title: a.title, fmt: a.fmt, content: '', char_count: a.char_count });
    });
    renderArtifactPicker();
  });
}

function deleteSession(id) {
  if (!window.confirm('确定删除该会话及其消息吗？此操作不可撤销。')) return;
  API.deleteSession(id).then(function (res) {
    if (!res.ok) { toast('删除失败：' + res.error, 'error'); return; }
    if (state.activeSessionId === id) state.activeSessionId = null;
    toast('会话已删除。', 'ok');
    refreshSessions();
  });
}

/** 渲染文献库列表（支持关键词过滤与排序）。 */
var libraryCache = [];
function refreshLibrary() {
  var box = $('libraryList');
  var sort = $('librarySort');
  return API.papers({ limit: 200, order_by: (sort && sort.value) || 'created_desc' }).then(function (res) {
    if (!box) return;
    if (!res.ok) {
      showPanelError(box, '文献库加载失败：' + res.error, refreshLibrary, '重试');
      return;
    }
    libraryCache = (res.data && res.data.items) || [];
    renderLibraryList();
  });
}

function renderLibraryList() {
  var box = $('libraryList');
  var count = $('libraryCount');
  if (!box) return;
  var filterInput = $('libraryFilter');
  var q = (filterInput && filterInput.value || '').trim().toLowerCase();
  var items = libraryCache;
  if (q) {
    items = items.filter(function (p) {
      var hay = [p.title, p.journal, (p.authors || []).join(' '), p.doi, p.pmid].join(' ').toLowerCase();
      return hay.indexOf(q) !== -1;
    });
  }
  if (count) count.textContent = q ? (items.length + ' / ' + libraryCache.length) : String(libraryCache.length);
  clear(box);
  if (!items.length) {
    box.appendChild(el('p', {
      class: 'empty-hint',
      text: libraryCache.length ? '没有匹配的文献。' : '文献库为空。先执行一次联网检索并勾选「入库」。'
    }));
    return;
  }
  items.slice(0, 300).forEach(function (p) {
    var key = paperKey(p);
    box.appendChild(el('button', {
      type: 'button',
      class: 'lib-item' + (state.selected[key] ? ' is-selected' : ''),
      title: p.title || '',
      onclick: function () { focusPaper(p); }
    }, [
      el('span', { class: 'lib-title', text: p.title || '（无标题）' }),
      el('span', { class: 'lib-meta' }, [
        el('span', { text: p.pub_year ? String(p.pub_year) : '年份未知' }),
        el('span', { text: p.source_label || sourceLabel(p.source) }),
        p.cited_by_count ? el('span', { text: '被引 ' + p.cited_by_count }) : null,
        p.is_open_access ? el('span', { class: 'chip chip-teal', text: 'OA' }) : null
      ])
    ]));
  });
}

/* ==========================================================================
 * §8. 检索面板
 * --------------------------------------------------------------------------
 * 联网检索：POST /api/search/live（逐源状态行显式呈现部分失败）
 * 本地检索：POST /api/papers/search（融合分数 + matched_by + 排名明细）
 * ======================================================================== */

function bindSearchPanel() {
  var form = $('searchForm');
  if (form) {
    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      runSearch();
    });
  }
  var sync = $('syncTopicBtn');
  if (sync) {
    sync.addEventListener('click', function () {
      var topic = ($('topicInput').value || '').trim();
      if (isPlaceholderTopic(topic)) { toast('请先在顶部输入课题。', 'warn'); return; }
      $('searchQuery').value = topic;
      $('searchQuery').focus();
      scheduleQueryPreview();
    });
  }
  var toggle = $('searchToggle');
  var body = $('searchBody');
  if (toggle && body) {
    toggle.addEventListener('click', function () {
      var open = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
      body.hidden = open;
    });
  }
  // 检索模式切换（联网 / 本地知识库）
  Array.prototype.forEach.call(document.querySelectorAll('[data-search-mode]'), function (btn) {
    btn.addEventListener('click', function () { setSearchMode(btn.getAttribute('data-search-mode')); });
  });
  setSearchMode(state.searchMode, true);

  var yearFrom = $('searchYearFrom');
  if (yearFrom) yearFrom.value = '2015';

  bindQueryBuilder();
}

/* --------------------------------------------------------------------------
 * 组合检索（查询构造器）
 *
 * 用户在检索框里写的是"自然语言 + 分隔符"（空格/逗号 = AND、| 或 ; = OR、
 * - 或 NOT = 排除、"..." = 短语）。这里把输入交给后端 /api/query/preview
 * 做解析并实时回显，解析逻辑与真正检索**同源**，所以预览所见即实际所发。
 *
 * 输入做 350ms 防抖，避免每敲一个字就打一次接口。
 * ------------------------------------------------------------------------ */
var queryPreviewTimer = null;

function bindQueryBuilder() {
  var input = $('searchQuery');
  if (input) {
    input.addEventListener('input', function () { scheduleQueryPreview(); });
  }

  Array.prototype.forEach.call(document.querySelectorAll('[data-insert]'), function (btn) {
    btn.addEventListener('click', function () {
      var text = btn.getAttribute('data-insert') || '';
      insertIntoQuery(text, false);
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-insert-quote]'), function (btn) {
    btn.addEventListener('click', function () { insertIntoQuery('""', true); });
  });

  var clear = $('queryClearBtn');
  if (clear) {
    clear.addEventListener('click', function () {
      var box = $('searchQuery');
      if (!box) return;
      box.value = '';
      box.focus();
      scheduleQueryPreview();
    });
  }

  var toggle = $('queryDetailToggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var detail = $('queryDetail');
      if (!detail) return;
      var open = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
      detail.hidden = open;
    });
  }

  // 面板上的"数据源：…"改成按钮，点击直接打开顶部同一个下拉，避免两处状态不同步
  var note = $('searchSourceNote');
  if (note) {
    note.addEventListener('click', function () {
      var btn = $('sourceSelectBtn');
      if (btn) { btn.click(); btn.focus(); }
    });
  }
}

/** 把片段插入检索框；caretInside 为 true 时把光标放到引号中间。 */
function insertIntoQuery(text, caretInside) {
  var box = $('searchQuery');
  if (!box) return;
  var start = box.selectionStart === null ? box.value.length : box.selectionStart;
  var end = box.selectionEnd === null ? box.value.length : box.selectionEnd;
  var before = box.value.slice(0, start);
  var after = box.value.slice(end);
  box.value = before + text + after;
  var caret = start + text.length - (caretInside ? 1 : 0);
  box.setSelectionRange(caret, caret);
  box.focus();
  scheduleQueryPreview();
}

function scheduleQueryPreview() {
  if (queryPreviewTimer) window.clearTimeout(queryPreviewTimer);
  queryPreviewTimer = window.setTimeout(refreshQueryPreview, 350);
}

function refreshQueryPreview() {
  var box = $('searchQuery');
  var wrap = $('queryPreview');
  var target = $('queryPreviewText');
  if (!box || !wrap || !target) return;

  var value = (box.value || '').trim();
  // 本地知识库模式走的是 FTS 分词，不适用布尔语法，直接隐藏预览
  if (!value || state.searchMode !== 'live') {
    wrap.hidden = true;
    var detail = $('queryDetail');
    if (detail) detail.hidden = true;
    return;
  }

  API.queryPreview(value, selectedSources()).then(function (res) {
    if (!res.ok || !res.data || !res.data.parsed) {
      wrap.hidden = true;
      return;
    }
    var parsed = res.data.parsed;
    if (parsed.is_simple) {
      // 简单查询无需强调解析结果，省掉一行视觉噪音
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;
    target.textContent = parsed.description || '';

    var detailBox = $('queryDetail');
    if (detailBox && !detailBox.hidden) {
      renderQueryDetail(detailBox, res.data);
    }
  });
}

function renderQueryDetail(box, data) {
  box.textContent = '';
  var perSource = data.per_source || {};
  Object.keys(perSource).forEach(function (name) {
    var row = el('div', { class: 'query-detail-row' });
    row.appendChild(el('span', { class: 'query-detail-src', text: sourceLabel(name) }));
    row.appendChild(el('span', { class: 'query-detail-expr', text: perSource[name] || '—' }));
    box.appendChild(row);
  });
  if (!Object.keys(perSource).length) {
    box.appendChild(el('div', { class: 'query-detail-row', text: '未选择数据源。' }));
  }
}

function setSearchMode(mode, silent) {
  state.searchMode = mode === 'local' ? 'local' : 'live';
  lsSet(STORE_KEYS.searchMode, state.searchMode);
  Array.prototype.forEach.call(document.querySelectorAll('[data-search-mode]'), function (btn) {
    var active = btn.getAttribute('data-search-mode') === state.searchMode;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-mode]'), function (node) {
    node.hidden = node.getAttribute('data-mode') !== state.searchMode;
  });
  var text = $('searchSubmitText');
  if (text) text.textContent = state.searchMode === 'local' ? '检索本地知识库' : '开始联网检索';
  var note = $('searchSourceNote');
  if (note && state.searchMode === 'local') note.textContent = '数据源：本地知识库（BM25 + 向量混合检索）';
  else renderSourceNote();
  if (!silent) dbg('search mode', state.searchMode);
}

/** 当前选中的数据源（规范化、去重）。 */
function selectedSources() {
  return state.sources.map(normalizeSource).filter(function (v, i, a) { return v && a.indexOf(v) === i; });
}

/** 数据源下拉：依据 /api/health 的 sources 渲染，缺失时用内置清单兜底。 */
function renderSourceMenu() {
  var body = $('sourceMenuBody');
  if (!body) return;
  var meta = {};
  if (state.health && Array.isArray(state.health.sources)) {
    state.health.sources.forEach(function (s) {
      if (s && s.name) meta[normalizeSource(s.name)] = s;
    });
  }
  clear(body);
  SOURCE_DEFS.forEach(function (def) {
    var m = meta[def.key];
    var disabled = m && m.enabled === false;
    var checked = state.sources.indexOf(def.key) !== -1;
    var cb = el('input', { type: 'checkbox', value: def.key, checked: checked });
    if (disabled) cb.disabled = true;
    cb.addEventListener('change', function () {
      var list = selectedSources();
      var idx = list.indexOf(def.key);
      if (cb.checked && idx === -1) list.push(def.key);
      if (!cb.checked && idx !== -1) list.splice(idx, 1);
      state.sources = list;
      lsSet(STORE_KEYS.sources, state.sources);
      updateSourceButton();
      renderSourceNote();
    });
    body.appendChild(el('label', { class: 'source-row' + (disabled ? ' is-disabled' : '') }, [
      cb,
      el('span', { class: 'source-row-name', text: def.label }),
      el('span', { class: 'source-row-label', text: disabled ? '已禁用' : (m && m.has_api_key ? '已配置密钥' : '') })
    ]));
  });
  updateSourceButton();
  renderSourceNote();
}

function updateSourceButton() {
  var text = $('sourceSelectText');
  if (!text) return;
  var n = selectedSources().length;
  text.textContent = n === 0 ? '未选择数据源' : (n === SOURCE_DEFS.length ? '全部数据源' : '数据源 · ' + n);
  var btn = $('sourceSelectBtn');
  if (btn) btn.title = n ? selectedSources().map(sourceLabel).join('、') : '未选择任何数据源';
}

function renderSourceNote() {
  var note = $('searchSourceNote');
  if (!note || state.searchMode === 'local') return;
  var list = selectedSources();
  note.textContent = '数据源：' + (list.length ? list.map(sourceLabel).join('、') : '未选择');
  // 数据源变了，各库的检索式预览也要跟着变（不同库的布尔支持不一样）
  if (typeof scheduleQueryPreview === 'function') scheduleQueryPreview();
}

/** 执行检索（依据当前模式分派到两个端点）。 */
function runSearch() {
  var query = ($('searchQuery').value || '').trim();
  if (!query) { toast('请输入检索关键词。', 'warn'); $('searchQuery').focus(); return; }

  var yearFrom = intOrNull($('searchYearFrom').value);
  var yearTo = intOrNull($('searchYearTo').value);
  if (yearFrom && yearTo && yearFrom > yearTo) {
    toast('起始年不能晚于截止年。', 'warn');
    return;
  }

  if (state.searchMode === 'local') searchLocal(query, yearFrom, yearTo);
  else searchLive(query, yearFrom, yearTo);
}

function intOrNull(v) {
  if (v === '' || v === null || v === undefined) return null;
  var n = parseInt(v, 10);
  return isFinite(n) ? n : null;
}

/** 联网检索：POST /api/search/live。 */
function searchLive(query, yearFrom, yearTo) {
  var sources = selectedSources();
  if (!sources.length) { toast('请至少选择一个数据源。', 'warn'); return; }

  var payload = {
    query: query,
    sources: sources,
    per_source_limit: intOrNull($('searchPerSource').value) || 20,
    filters: {
      year_from: yearFrom,
      year_to: yearTo,
      open_access: !!($('searchOpenAccess') && $('searchOpenAccess').checked),
      sort: 'relevance'
    },
    save: !!($('searchSave') && $('searchSave').checked),
    embed: !!($('searchEmbed') && $('searchEmbed').checked)
  };

  setSearchBusy(true, '检索中…');
  clearSourceStatus();
  pushTool('search', '联网检索', '“' + query + '” · ' + sources.map(sourceLabel).join('、'));

  API.searchLive(payload).then(function (res) {
    setSearchBusy(false);
    if (!res.ok) {
      renderSearchFailure(res.error, function () { searchLive(query, yearFrom, yearTo); });
      pushTool('error', '检索失败', res.error);
      state.lastSearchOk = false;
      return;
    }
    var d = res.data || {};
    state.lastSearchOk = true;
    renderSourceStatus(d.sources || []);
    renderSearchSummary(d);
    var items = d.items || [];
    if (items.length) {
      upsertPapers(items, true);
      switchTab('papers');
    } else {
      toast('检索完成，但所有数据源都没有返回可用文献。', 'warn');
    }
    pushTool('search', '检索完成',
      '共 ' + (d.count === undefined ? items.length : d.count) + ' 条结果（原始 ' + (d.raw_count === undefined ? '—' : d.raw_count) + ' 条），耗时 ' + fmtMs(d.duration_ms),
      d.sources || []);
    if (d.saved) pushStatus('入库：新增 ' + (d.saved['new'] || 0) + ' 条，更新 ' + (d.saved.updated || 0) + ' 条。');
    if (d.embedded) pushStatus('向量化：' + (d.embedded.embedded || 0) + ' 条。');
    refreshLibrary();
    refreshHealth({ silent: true });
  });
}

/** 本地知识库混合检索：POST /api/papers/search。 */
function searchLocal(query, yearFrom, yearTo) {
  var filters = {};
  if (yearFrom) filters.year_from = yearFrom;
  if (yearTo) filters.year_to = yearTo;
  if ($('searchLocalOA') && $('searchLocalOA').checked) filters.open_access = true;

  var payload = { query: query, top_k: intOrNull($('searchTopK').value) || 20, filters: filters };

  setSearchBusy(true, '检索本地库…');
  clearSourceStatus();
  pushTool('search', '本地知识库检索', '“' + query + '”' + (Object.keys(filters).length ? ' · 过滤条件已应用' : ''));

  API.searchLocal(payload).then(function (res) {
    setSearchBusy(false);
    if (!res.ok) {
      renderSearchFailure(res.error, function () { searchLocal(query, yearFrom, yearTo); });
      pushTool('error', '本地检索失败', res.error);
      return;
    }
    var d = res.data || {};
    var items = d.items || [];
    var ok = el('div', { class: 'search-summary' }, [
      el('span', { class: 'chip chip-accent', text: '命中 ' + (d.count === undefined ? items.length : d.count) + ' 条' }),
      el('span', { text: '融合排序：BM25 + 向量（RRF）' })
    ]);
    var box = $('searchSummary');
    if (box) { box.hidden = false; clear(box); box.appendChild(ok); }

    if (!items.length) {
      toast('本地知识库中没有匹配文献。可先做一次联网检索入库。', 'warn');
    } else {
      upsertPapers(items, true);
      switchTab('papers');
      pushTool('search', '本地检索完成',
        '命中 ' + items.length + ' 条，最高分 ' + fmtScore(items[0].score),
        null, items.slice(0, 5).map(function (p) {
          return { label: p.citation_label || p.title, text: 'score ' + fmtScore(p.score) + ' · ' + (p.matched_by || '—') };
        }));
    }
  });
}

function fmtScore(v) {
  var n = Number(v);
  if (!isFinite(n)) return '—';
  return n.toFixed(4);
}

function setSearchBusy(busy, label) {
  var btn = $('searchSubmitBtn');
  var prog = $('searchProgress');
  if (btn) btn.disabled = busy;
  if (prog) {
    prog.hidden = !busy;
    if (busy && label) prog.textContent = label;
  }
}

/** 渲染单源状态行：✔ 条数 / ✖ 错误 / 耗时；部分失败在此显式可见。 */
function renderSourceStatus(sources) {
  var box = $('sourceStatus');
  if (!box) return;
  clear(box);
  if (!sources.length) { box.hidden = true; return; }
  box.hidden = false;
  sources.forEach(function (s) {
    var ok = s.ok !== false;
    var skipped = s.skipped === true;
    var cls = skipped ? 'is-skip' : (ok ? 'is-ok' : 'is-bad');
    var kids = [icon(skipped ? 'info' : (ok ? 'check' : 'x'), 'ic-xs')];
    kids.push(el('span', { class: 'src-chip-name', text: s.label || sourceLabel(s.name) }));
    if (skipped) {
      kids.push(el('span', { class: 'src-chip-count', text: '已跳过' }));
    } else if (ok) {
      kids.push(el('span', { class: 'src-chip-count', text: String(s.count === undefined ? 0 : s.count) + ' 条' }));
    } else {
      kids.push(el('span', { class: 'src-chip-err', title: s.error || '未知错误', text: '✖ ' + (s.error || '检索失败') }));
    }
    if (s.duration_ms !== undefined && s.duration_ms !== null) {
      kids.push(el('span', { class: 'src-chip-ms', text: fmtMs(s.duration_ms) }));
    }
    box.appendChild(el('span', { class: 'src-chip ' + cls }, kids));
  });

  var failed = sources.filter(function (s) { return s.ok === false && s.skipped !== true; });
  if (failed.length) {
    box.appendChild(el('span', {
      class: 'chip chip-warn',
      text: failed.length + ' 个数据源失败，其余结果仍然可用'
    }));
  }
}

function clearSourceStatus() {
  var box = $('sourceStatus');
  if (box) { clear(box); box.hidden = true; }
  var sum = $('searchSummary');
  if (sum) { clear(sum); sum.hidden = true; }
}

function renderSearchSummary(d) {
  var box = $('searchSummary');
  if (!box) return;
  clear(box);
  box.hidden = false;
  box.appendChild(el('span', { class: 'chip chip-accent', text: '入库结果 ' + (d.count === undefined ? 0 : d.count) + ' 条' }));
  box.appendChild(el('span', { text: '原始命中 ' + (d.raw_count === undefined ? '—' : d.raw_count) + ' 条 · 总耗时 ' + fmtMs(d.duration_ms) }));
  if (d.saved) box.appendChild(el('span', { class: 'chip', text: '新增 ' + (d.saved['new'] || 0) + ' · 更新 ' + (d.saved.updated || 0) }));
  if (d.embedded) box.appendChild(el('span', { class: 'chip', text: '向量化 ' + (d.embedded.embedded || 0) }));
}

function renderSearchFailure(errorText, retryFn) {
  var box = $('searchSummary');
  if (!box) return;
  clear(box);
  box.hidden = false;
  box.appendChild(noticeNode('error', '检索失败：' + errorText, [
    el('button', { class: 'btn btn-sm', type: 'button', text: '重试', onclick: retryFn })
  ]));
}

/* ==========================================================================
 * §9. 文献集合与文献卡片
 * --------------------------------------------------------------------------
 * 聚合来自 /api/search/live、/api/agent/stream（papers / search_result）
 * 与 /api/papers/search 的文献，统一渲染为可选中的卡片。
 * ======================================================================== */

/** 文献去重键：优先 paper_id，其次 dedup_key，最后标题指纹。 */
function paperKey(p) {
  if (!p) return 'unknown';
  if (p.paper_id !== undefined && p.paper_id !== null) return 'id:' + p.paper_id;
  if (p.dedup_key) return p.dedup_key;
  return 'title:' + String(p.title || '').slice(0, 80);
}

/** 合并文献到集合（按 key 去重，保留检索附加的评分字段）。 */
function upsertPapers(items, markNew) {
  if (!Array.isArray(items) || !items.length) return;
  var added = 0;
  items.forEach(function (p) {
    if (!p || typeof p !== 'object') return;
    if (!p.title && !p.paper_id) return;
    var key = paperKey(p);
    var existing = state.paperIndex[key];
    if (existing) {
      // 保留已有评分字段，补足新字段
      Object.keys(p).forEach(function (k) { existing[k] = p[k]; });
    } else {
      state.paperIndex[key] = p;
      state.papers.push(p);
      added++;
      if (markNew) p.__isNew = true;
    }
  });
  renderPapers();
  var badge = $('tabBadgePapers');
  if (badge) {
    badge.textContent = String(state.papers.length);
    badge.hidden = state.papers.length === 0;
  }
  if (added) dbg('papers +' + added + ' → ' + state.papers.length);
}

function renderPapers() {
  var box = $('papersList');
  if (!box) return;
  clear(box);
  if (!state.papers.length) {
    box.appendChild(el('p', { class: 'empty-hint', text: '尚无文献。先执行一次检索，或启动研究工作流。' }));
    updateSelectedCount();
    return;
  }
  state.papers.forEach(function (p) {
    box.appendChild(paperCard(p));
    p.__isNew = false;
  });
  updateSelectedCount();
}

/** 渲染单张文献卡片（含评分/排名明细与选择框）。 */
function paperCard(p) {
  var key = paperKey(p);
  var selected = !!state.selected[key];

  var checkbox = el('input', {
    type: 'checkbox', class: 'paper-check', checked: selected,
    title: '选中以便生成引用或导出', 'aria-label': '选择文献'
  });
  checkbox.addEventListener('change', function () {
    if (checkbox.checked) state.selected[key] = true;
    else delete state.selected[key];
    var card = checkbox.closest ? checkbox.closest('.paper-card') : null;
    if (card) card.classList.toggle('is-selected', checkbox.checked);
    updateSelectedCount();
    renderLibraryList();
  });

  var meta = el('div', { class: 'paper-meta' }, [
    p.pub_year ? el('span', { text: String(p.pub_year) }) : el('span', { text: '年份未知' }),
    p.journal ? el('span', { class: 'paper-journal', text: p.journal }) : null,
    el('span', { text: p.source_label || sourceLabel(p.source) }),
    p.cited_by_count ? el('span', { text: '被引 ' + p.cited_by_count }) : null,
    p.publication_type ? el('span', { text: p.publication_type }) : null,
    p.is_open_access ? el('span', { class: 'chip chip-teal', text: '开放获取' }) : null,
    p.matched_by ? el('span', { class: 'chip ' + matchedByClass(p.matched_by), text: matchedByLabel(p.matched_by) }) : null,
    (p.score !== undefined && p.score !== null) ? el('span', { class: 'chip chip-accent', text: 'score ' + fmtScore(p.score) }) : null
  ]);

  var rankDetail = null;
  if (p.fts_rank !== undefined || p.vector_rank !== undefined || p.fts_score !== undefined || p.vector_distance !== undefined) {
    rankDetail = el('div', { class: 'rank-detail paper-meta' }, [
      el('span', { class: 'field-label', text: '排名明细' }),
      p.fts_rank !== undefined && p.fts_rank !== null ? el('span', { class: 'chip', text: 'BM25 #' + p.fts_rank }) : el('span', { class: 'chip chip-quiet', text: 'BM25 未命中' }),
      p.vector_rank !== undefined && p.vector_rank !== null ? el('span', { class: 'chip', text: '向量 #' + p.vector_rank }) : el('span', { class: 'chip chip-quiet', text: '向量未命中' }),
      p.fts_score !== undefined && p.fts_score !== null ? el('span', { class: 'chip chip-quiet', text: 'bm25=' + p.fts_score }) : null,
      p.vector_distance !== undefined && p.vector_distance !== null ? el('span', { class: 'chip chip-quiet', text: 'dist=' + p.vector_distance }) : null
    ]);
  }

  var abstractText = p.abstract || '';
  var abstract = abstractText
    ? el('p', { class: 'paper-abstract is-clamped', text: truncate(abstractText, 420) })
    : el('p', { class: 'empty-hint', text: '（无摘要）' });
  if (abstractText && abstractText.length > 420) {
    abstract.style.cursor = 'pointer';
    abstract.title = '点击展开/收起摘要';
    abstract.addEventListener('click', function () { abstract.classList.toggle('is-clamped'); });
  }

  var ids = el('div', { class: 'paper-ids' }, [
    p.pmid ? el('span', { text: 'PMID ' + p.pmid }) : null,
    p.doi ? el('span', { text: 'DOI ' + p.doi }) : null,
    p.citation_label ? el('span', { text: p.citation_label }) : null
  ]);

  var libraryBtn = buildLibraryButton(p);
  var actions = el('div', { class: 'paper-foot-actions' }, [
    p.url ? el('a', { class: 'btn btn-sm', href: p.url, target: '_blank', rel: 'noopener noreferrer', title: '在浏览器中打开原文' }, [icon('external', 'ic-xs'), el('span', { text: '原文' })]) : null,
    libraryBtn,
    el('button', {
      type: 'button', class: 'btn btn-sm', title: '查看后端完整元数据',
      onclick: function () { openPaperDetail(p, this); }
    }, [icon('info', 'ic-xs'), el('span', { text: '详情' })])
  ]);

  var foot = el('div', { class: 'paper-foot' }, [ids, actions]);
  var detail = el('div', { class: 'paper-detail', hidden: true });

  var card = el('article', {
    class: 'paper-card' + (selected ? ' is-selected' : '') + (p.__isNew ? ' is-new' : ''),
    'data-paper-key': key
  }, [
    el('div', { class: 'paper-top' }, [
      checkbox,
      el('h4', { class: 'paper-title', text: p.title || '（无标题）' })
    ]),
    el('p', { class: 'paper-authors', text: p.short_authors || (p.authors || []).join(', ') || '作者信息缺失' }),
    meta,
    abstract,
    rankDetail,
    (p.mesh_terms && p.mesh_terms.length) ? el('div', { class: 'taglist paper-meta' }, p.mesh_terms.slice(0, 8).map(function (t) {
      return el('span', { class: 'tag', text: t });
    })) : null,
    foot,
    detail
  ]);
  card.__detail = detail;
  return card;
}

/**
 * 「通过图书馆获取全文」按钮。
 *
 * 这是订阅资源**唯一合规**的用法：不下载、不抓取，只生成一个走学校
 * 链接解析器 / 校外访问代理的链接，用户点开后用自己的会话阅读。
 * 链接模板由用户在设置里填写（各校解析器地址不同，不能猜）。
 */
function buildLibraryButton(p) {
  var s = state.settings || {};
  if (!s.libraryEnabled || !p.doi) return null;
  var url = libraryUrl(p.doi, s);
  if (!url) return null;
  return el('a', {
    class: 'btn btn-sm',
    href: url,
    target: '_blank',
    rel: 'noopener noreferrer',
    title: '经' + (s.libraryName || '图书馆') + '链接解析器打开（用你自己的账号查看订阅全文）'
  }, [icon('external', 'ic-xs'), el('span', { text: s.libraryName || '图书馆全文' })]);
}

/** 按模板拼出图书馆链接。支持 {doi} / {url} 占位符。 */
function libraryUrl(doi, settings) {
  var s = settings || state.settings || {};
  var clean = String(doi || '').trim();
  if (!clean) return '';
  var doiUrl = 'https://doi.org/' + clean;
  var template = '';
  if (s.libraryOpenUrl) {
    template = String(s.libraryOpenUrl).trim();
    if (template.indexOf('{doi}') === -1 && template.indexOf('{url}') === -1) {
      // 用户只填了解析器基地址：按 OpenURL 标准补上 DOI 参数
      template = template.replace(/\?*$/, '') + '?url_ver=Z39.88-2004'
        + '&rft_val_fmt=info:ofi/fmt:kev:mtx:journal&genre=article&id=doi:{doi}';
    }
  } else if (s.libraryProxyPrefix) {
    template = String(s.libraryProxyPrefix).trim() + '{url}';
  }
  if (!template) return '';
  return template.split('{doi}').join(encodeURIComponent(clean))
                 .split('{url}').join(encodeURIComponent(doiUrl));
}

/** 导入题录文件：读文本 → POST /api/import。 */
function importFiles(files) {
  if (!files || !files.length) return;
  var list = Array.prototype.slice.call(files);
  var hint = $('importHint');
  if (hint) {
    hint.hidden = false;
    hint.textContent = '正在导入 ' + list.length + ' 个文件…';
    hint.classList.remove('empty-hint-warn');
  }
  setStatus('正在导入 ' + list.length + ' 个题录文件…');

  var totals = { parsed: 0, unique: 0, created: 0, merged: 0, failed: 0, embedded: 0 };
  var problems = [];
  var formats = [];

  var chain = Promise.resolve();
  list.forEach(function (file) {
    chain = chain.then(function () {
      return new Promise(function (resolve) {
        var reader = new FileReader();
        reader.onerror = function () {
          problems.push(file.name + '：读取失败');
          resolve();
        };
        reader.onload = function () {
          API.importPapers({
            content: String(reader.result || ''),
            filename: file.name,
            source: 'import',
            embed: true
          }).then(function (res) {
            if (!res.ok) {
              problems.push(file.name + '：' + res.error);
              return;
            }
            var d = res.data || {};
            totals.parsed += d.parsed || 0;
            totals.unique += d.unique || 0;
            totals.created += d.created || 0;
            totals.merged += d.merged || 0;
            totals.failed += d.failed || 0;
            if (d.embedded && d.embedded.embedded) totals.embedded += d.embedded.embedded;
            if (d.format && formats.indexOf(d.format) === -1) formats.push(d.format);
            (d.errors || []).forEach(function (m) { problems.push(file.name + '：' + m); });
          }).then(resolve, resolve);
        };
        reader.readAsText(file, 'utf-8');
      });
    });
  });

  chain.then(function () {
    var msg = '导入完成：解析 ' + totals.parsed + ' 条 → 去重后 ' + totals.unique
      + ' 条，新增 ' + totals.created + ' 篇'
      + (totals.merged ? '，合并 ' + totals.merged + ' 篇' : '')
      + (totals.embedded ? '，生成向量 ' + totals.embedded + ' 条' : '')
      + '。';
    if (hint) {
      hint.hidden = false;
      hint.textContent = msg + (formats.length ? '（格式：' + formats.join('、') + '）' : '');
      hint.classList.toggle('empty-hint-warn', totals.created === 0 && problems.length > 0);
    }
    if (problems.length) {
      problems.slice(0, 3).forEach(function (m) { pushError('导入问题：' + m); });
    }
    toast(msg, totals.created || totals.merged ? 'ok' : 'warn', 8000);
    pushTool('phase', '导入题录文件', msg
      + (formats.length ? ' 格式：' + formats.join('、') : '')
      + (problems.length ? ' · ' + problems.length + ' 个问题' : ''));
    setStatus(msg);
    refreshLibrary();
  });
}

function matchedByLabel(v) {  var s = String(v || '');
  if (s === 'bm25') return 'BM25 命中';
  if (s === 'vector') return '向量命中';
  if (s === 'bm25+vector') return 'BM25 + 向量';
  return s || '—';
}
function matchedByClass(v) {
  var s = String(v || '');
  if (s === 'bm25+vector') return 'chip-accent';
  if (s === 'vector') return 'chip-teal';
  return '';
}

/** 通过 GET /api/papers/{id} 拉取完整元数据并就地展开。 */
function openPaperDetail(p, btn) {
  var card = btn && btn.closest ? btn.closest('.paper-card') : null;
  var detail = card && card.__detail;
  if (!detail) return;
  if (!detail.hidden) { detail.hidden = true; return; }
  if (p.paper_id === undefined || p.paper_id === null) {
    detail.hidden = false;
    clear(detail);
    detail.appendChild(noticeNode('warn', '该文献尚未入库（无 paper_id），因此无法读取完整元数据。请在检索时勾选「入库」。'));
    return;
  }
  detail.hidden = false;
  clear(detail);
  detail.appendChild(el('p', { class: 'empty-hint', text: '正在读取元数据…' }));
  API.paper(p.paper_id).then(function (res) {
    clear(detail);
    if (!res.ok) {
      detail.appendChild(noticeNode('error', '元数据读取失败：' + res.error));
      return;
    }
    var d = res.data || {};
    var dl = el('dl', { class: 'plan-kv' });
    [['标题', d.title], ['作者', (d.authors || []).join(', ')], ['期刊', d.journal],
     ['年份', d.pub_year], ['卷/期/页', [d.volume, d.issue, d.pages].filter(Boolean).join(' / ')],
     ['发表类型', d.publication_type], ['语种', d.language], ['PMID', d.pmid], ['PMCID', d.pmcid],
     ['DOI', d.doi], ['去重键', d.dedup_key], ['入库时间', d.created_at], ['更新时间', d.updated_at],
     ['MeSH', (d.mesh_terms || []).join('; ')], ['关键词', (d.keywords || []).join('; ')]
    ].forEach(function (pair) {
      if (pair[1] === undefined || pair[1] === null || pair[1] === '') return;
      dl.appendChild(el('dt', { text: pair[0] }));
      dl.appendChild(el('dd', { text: String(pair[1]) }));
    });
    detail.appendChild(dl);
    if (d.abstract) detail.appendChild(el('p', { class: 'paper-abstract', text: d.abstract }));
  });
}

/** 在右栏定位并高亮某篇文献（来自左栏文献库点击）。 */
function focusPaper(p) {
  var key = paperKey(p);
  if (!state.paperIndex[key]) upsertPapers([p], false);
  switchTab('papers');
  var box = $('papersList');
  if (!box) return;
  var card = box.querySelector('[data-paper-key="' + cssEscape(key) + '"]');
  if (card) {
    card.scrollIntoView({ block: 'center', behavior: 'smooth' });
    card.classList.add('is-new');
    window.setTimeout(function () { card.classList.remove('is-new'); }, 1200);
  }
}

/** CSS 属性选择器转义（key 含冒号等字符）。 */
function cssEscape(s) {
  return String(s).replace(/["\\]/g, '\\$&');
}

function updateSelectedCount() {
  var n = Object.keys(state.selected).length;
  var node = $('selectedCount');
  if (node) node.textContent = '已选 ' + n + ' 篇';
  var all = $('selectAllPapers');
  if (all) {
    all.checked = state.papers.length > 0 && n >= state.papers.length;
    all.indeterminate = n > 0 && n < state.papers.length;
  }
}

/** 已选文献的数值 ID 列表（用于 /api/cite 与 /api/export）。 */
function selectedPaperIds() {
  var ids = [];
  var missing = 0;
  Object.keys(state.selected).forEach(function (key) {
    var p = state.paperIndex[key];
    if (!p) return;
    if (p.paper_id === undefined || p.paper_id === null) { missing++; return; }
    var n = parseInt(p.paper_id, 10);
    if (isFinite(n)) ids.push(n);
  });
  return { ids: ids, missing: missing };
}

/* ==========================================================================
 * §10. 右栏标签页 / 引用 / 导出 / 草稿
 * --------------------------------------------------------------------------
 * 引用：POST /api/cite      导出：POST /api/export
 * 产物：GET /api/artifacts/{id}（会话产物列表见 §7）
 * ======================================================================== */

function switchTab(name) {
  state.activeTab = name;
  lsSet(STORE_KEYS.tab, name);
  Array.prototype.forEach.call(document.querySelectorAll('.tabs .tab'), function (t) {
    var active = t.getAttribute('data-tab') === name;
    t.classList.toggle('is-active', active);
    t.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  Array.prototype.forEach.call(document.querySelectorAll('.tabpane'), function (pn) {
    pn.classList.toggle('is-active', pn.getAttribute('data-pane') === name);
  });
  // 草稿/论文页隐藏批量选择条，其余页面显示
  var bar = $('artifactBar');
  if (bar) bar.hidden = (name === 'draft' || name === 'manuscript');
}

function bindTabs() {
  Array.prototype.forEach.call(document.querySelectorAll('.tabs .tab'), function (t) {
    t.addEventListener('click', function () { switchTab(t.getAttribute('data-tab')); });
  });
  switchTab(state.activeTab || 'papers');
}

/** 生成引用：POST /api/cite。 */
function generateCitations() {
  var sel = selectedPaperIds();
  if (!sel.ids.length) {
    toast(sel.missing
      ? '所选文献尚未入库（无 paper_id），无法生成引用。请在检索时勾选「入库」。'
      : '请先在「文献卡片」中勾选文献。', 'warn');
    return;
  }
  var style = state.citeStyle;
  var btn = $('generateCiteBtn');
  if (btn) btn.disabled = true;

  API.cite({ paper_ids: sel.ids, style: style, format: 'list' }).then(function (res) {
    if (btn) btn.disabled = false;
    if (!res.ok) {
      var box = $('citationsList');
      showPanelError(box, '引用生成失败：' + res.error, generateCitations, '重试');
      switchTab('citations');
      toast('引用生成失败：' + truncate(res.error, 100), 'error');
      return;
    }
    var d = res.data || {};
    renderCitations(d, style);
    switchTab('citations');
    pushTool('done', '生成引用', (d.label || style) + ' · ' + sel.ids.length + ' 条');
    toast('已生成 ' + sel.ids.length + ' 条引用（' + (d.label || style) + '）。', 'ok');
    if (sel.missing) toast('另有 ' + sel.missing + ' 篇未入库文献已跳过。', 'warn');
  });
}

/** 把 /api/cite 的纯文本结果拆分为可定位的条目。 */
function renderCitations(data, style) {
  var box = $('citationsList');
  if (!box) return;
  clear(box);
  var content = String(data && data.content ? data.content : '');
  var order = selectedPaperIds().ids;
  var isMono = style === 'bibtex' || style === 'ris';

  // BibTeX / RIS 以空行（BibTeX）或 TY 行（RIS）分条；其余样式按行分条。
  var chunks;
  if (style === 'bibtex') chunks = content.split(/\n\s*\n/);
  else if (style === 'ris') chunks = content.split(/\n(?=TY\s*-)/);
  else chunks = content.split(/\n/);
  chunks = chunks.map(function (c) { return c.replace(/\s+$/, ''); }).filter(function (c) { return c.length; });

  if (!chunks.length) {
    box.appendChild(el('p', { class: 'empty-hint', text: '后端返回的引用内容为空。' }));
    return;
  }

  state.citations = [];
  chunks.forEach(function (text, i) {
    var index = i + 1;
    var paperKeyForIndex = null;
    if (order[i] !== undefined) {
      paperKeyForIndex = 'id:' + order[i];
    }
    state.citations.push({ index: index, text: text, paperKey: paperKeyForIndex });
    box.appendChild(el('div', {
      class: 'cite-item', 'data-cite-index': String(index),
      onclick: function () { highlightCitation(index); }
    }, [
      el('span', { class: 'cite-index', text: isMono ? String(index) : '[' + index + ']' }),
      el('div', { class: 'cite-text' + (isMono ? ' cite-text-mono' : ''), text: text })
    ]));
  });

  var toolbar = $('citeStyleName');
  if (toolbar) toolbar.textContent = (data && data.label) || updateCiteStyleLabel();
}

/** 引用标记点击：滚动到对应条目并高亮。 */
function highlightCitation(index) {
  switchTab('citations');
  var box = $('citationsList');
  if (!box) return;
  var node = box.querySelector('[data-cite-index="' + index + '"]');
  if (!node) {
    toast('引用列表中还没有第 ' + index + ' 条。请先生成引用，或该编号超出当前列表范围。', 'warn');
    return;
  }
  node.scrollIntoView({ block: 'center', behavior: 'smooth' });
  Array.prototype.forEach.call(box.querySelectorAll('.cite-item.is-target'), function (n) { n.classList.remove('is-target'); });
  node.classList.add('is-target');
  window.setTimeout(function () { node.classList.remove('is-target'); }, 2600);
}

/** 导出：POST /api/export，并展示后端返回的文件路径。 */
function exportSelection() {
  var sel = selectedPaperIds();
  if (!sel.ids.length) {
    toast(sel.missing
      ? '所选文献尚未入库（无 paper_id），无法导出。'
      : '请先在「文献卡片」中勾选文献。', 'warn');
    return;
  }
  var format = ($('exportFormat') && $('exportFormat').value) || 'bibtex';
  var name = ($('topicInput').value || '').trim() || 'medscholar导出';
  var btn = $('exportBtn');
  if (btn) btn.disabled = true;

  API.exportPapers({ paper_ids: sel.ids, format: format, name: name }).then(function (res) {
    if (btn) btn.disabled = false;
    if (!res.ok) {
      toast('导出失败：' + truncate(res.error, 140), 'error');
      pushError('导出失败：' + res.error);
      return;
    }
    var d = res.data || {};
    var box = $('citationsList');
    if (box && state.activeTab === 'citations') {
      box.insertBefore(noticeNode('info', '已导出 ' + sel.ids.length + ' 篇文献（' + (d.format || format) + '），文件路径：' + (d.path || '未返回路径')), box.firstChild);
    }
    pushTool('done', '导出成功', (d.format || format) + ' · ' + (d.bytes === undefined ? '' : d.bytes + ' 字节') + (d.path ? ' · ' + d.path : ''));
    toast('已导出到：' + (d.path || '（后端未返回路径）'), 'ok', 9000);
  });
}

/** 渲染产物选择器并切换草稿内容。 */
function renderArtifactPicker() {
  var picker = $('draftPicker');
  var bar = $('draftToolbar');
  if (!picker || !bar) return;
  updateDraftEmptyHint();
  if (!state.artifacts.length) { bar.hidden = true; return; }
  bar.hidden = false;
  clear(picker);
  state.artifacts.forEach(function (a, i) {
    picker.appendChild(el('option', { value: String(i), text: a.title || ('产物 #' + a.id) }));
  });
  picker.value = String(state.activeArtifact);
}

function showArtifact(index) {
  var a = state.artifacts[index];
  if (!a) return;
  state.activeArtifact = index;
  var content = $('draftContent');
  if (!content) return;
  if (a.content) {
    setMarkdown(content, a.content);
  } else if (a.id !== undefined && a.id !== null) {
    clear(content);
    content.appendChild(el('p', { class: 'empty-hint', text: '正在读取产物正文…' }));
    API.artifact(a.id).then(function (res) {
      if (!res.ok) {
        showPanelError(content, '产物读取失败：' + res.error);
        return;
      }
      var d = res.data || {};
      a.content = d.content || '';
      a.title = d.title || a.title;
      a.fmt = d.fmt || a.fmt;
      setMarkdown(content, a.content || '（产物内容为空）');
    });
  } else {
    setMarkdown(content, '（该产物暂无正文）');
  }
  renderArtifactPicker();
}

/** 记录一个 Agent 产物（来自 SSE artifact 事件）。 */
function addArtifact(a) {
  if (!a) return;
  var id = a.artifact_id !== undefined ? a.artifact_id : a.id;
  var exists = -1;
  for (var i = 0; i < state.artifacts.length; i++) {
    if (String(state.artifacts[i].id) === String(id)) { exists = i; break; }
  }
  var item = { id: id, title: a.title || '综述草稿', fmt: a.fmt || 'markdown', content: a.content || '' };
  if (exists >= 0) state.artifacts[exists] = item;
  else state.artifacts.push(item);
  showArtifact(state.artifacts.length - 1);
  renderArtifactPicker();
}

/* ==========================================================================
 * §11. 对话视口与消息渲染
 * ======================================================================== */

function hideWelcome() {
  var w = $('welcomeBlock');
  if (w && w.parentNode) w.parentNode.removeChild(w);
}

function scrollChatToEnd(force) {
  var box = $('messageStream');
  if (!box) return;
  var nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 160;
  if (force || nearBottom) box.scrollTop = box.scrollHeight;
}

/** 追加一条消息气泡。role: user | agent | system | error | status */
function pushMessage(role, text, timeText, options) {
  var opts = options || {};
  hideWelcome();
  var box = $('messageStream');
  if (!box) return null;

  if (role === 'status') {
    var line = el('div', { class: 'msg msg-status' }, [
      el('div', { class: 'msg-body', text: text })
    ]);
    box.appendChild(line);
    scrollChatToEnd(true);
    return line;
  }

  var roleNames = { user: '你', agent: 'MedScholar', system: '系统', error: '错误' };
  var body = el('div', { class: 'msg-body' });
  if (opts.markdown) body.appendChild(renderMarkdown(text));
  else body.textContent = text;

  var node = el('div', { class: 'msg msg-' + role }, [
    el('div', { class: 'msg-head' }, [
      el('span', { class: 'msg-role', text: roleNames[role] || role }),
      el('span', { class: 'msg-time', text: timeText || clockNow() })
    ]),
    body
  ]);
  // AI 的每一条输出都可以被点赞/否定/质疑 —— 这是学习闭环的入口。
  // 注意：流式消息在创建时正文还是空的，所以要**点击时**从 DOM 取当前正文，
  // 否则反馈里带回去的原文永远是空串。
  if (role === 'agent' || role === 'error') {
    node.appendChild(buildFeedbackBar(function () {
      return (body.textContent || text || '').trim();
    }, node));
  }
  box.appendChild(node);
  scrollChatToEnd(true);
  return node;
}

function pushStatus(text) { return pushMessage('status', text); }
function pushError(text) { return pushMessage('error', text); }

/* ========================================================================
 * §14. 反馈、质疑与学习闭环
 *
 * 设计要点：**反馈必须真的改变系统后面的行为**，否则就只是装饰。
 *   赞/踩      → 构成 DPO 偏好对，可导出成训练数据（离线生效）
 *   质疑+纠错  → 立刻成为后续生成的"纠错记忆"（在线生效，不需要训练）
 * 因此界面上的"质疑"要求用户尽量写出**正确说法**，只点踩不写内容的
 * 反馈价值有限，文案里也如实说明这一点。
 * ======================================================================== */

var FEEDBACK_CATEGORIES = [
  { key: 'citation',  label: '引用错误（编号/出处对不上）' },
  { key: 'fact',      label: '事实错误（结论说反、数据不准）' },
  { key: 'omission',  label: '遗漏重要文献或结论' },
  { key: 'overreach', label: '过度推断（证据不支持）' },
  { key: 'structure', label: '结构或逻辑问题' },
  { key: 'style',     label: '文风与表述' },
  { key: 'data',      label: '与我提供的数据不一致' },
  { key: 'other',     label: '其它' }
];

/** 构造一条消息下方的反馈工具条。getText() 返回被评价的原文。 */
function buildFeedbackBar(getText, hostNode) {
  var bar = el('div', { class: 'fb-bar' });
  var upBtn = el('button', { type: 'button', class: 'fb-btn', title: '有帮助' }, [
    icon('check', 'ic-xs'), el('span', { text: '有用' })
  ]);
  var downBtn = el('button', { type: 'button', class: 'fb-btn', title: '没帮助' }, [
    icon('x', 'ic-xs'), el('span', { text: '没用' })
  ]);
  var challengeBtn = el('button', { type: 'button', class: 'fb-btn fb-challenge', title: '指出错误并给出正确说法' }, [
    icon('alert', 'ic-xs'), el('span', { text: '质疑 / 纠错' })
  ]);
  var status = el('span', { class: 'fb-status' });

  bar.appendChild(upBtn);
  bar.appendChild(downBtn);
  bar.appendChild(challengeBtn);
  bar.appendChild(status);

  function disableAll() {
    [upBtn, downBtn, challengeBtn].forEach(function (b) { b.disabled = true; });
  }

  function quick(verdict, button) {
    submitFeedback({
      verdict: verdict,
      target_type: 'message',
      quoted_text: truncate(String(getText() || ''), 800),
      topic: ($('topicInput') && $('topicInput').value || '').trim(),
      run_id: (state.run && state.run.id) || '',
      session_id: state.activeSessionId || null
    }).then(function (res) {
      if (res.ok) {
        button.classList.add('is-on');
        status.textContent = res.data && res.data.message || '已记录';
      } else {
        status.textContent = '提交失败：' + res.error;
      }
    });
  }

  upBtn.addEventListener('click', function () { quick('up', upBtn); });
  downBtn.addEventListener('click', function () { quick('down', downBtn); });
  challengeBtn.addEventListener('click', function () {
    if (bar.querySelector('.fb-panel')) return;
    var category = el('select', { class: 'input input-sm input-select' });
    FEEDBACK_CATEGORIES.forEach(function (item) {
      category.appendChild(el('option', { value: item.key, text: item.label }));
    });
    var quoted = el('textarea', {
      class: 'input input-sm', rows: 2,
      placeholder: '你觉得有问题的原文片段（可留空，默认取整条输出）'
    });
    quoted.value = truncate(String(getText() || ''), 400);
    var comment = el('textarea', {
      class: 'input input-sm', rows: 2,
      placeholder: '说明问题出在哪（例如：这条结论与原文相反）'
    });
    var corrected = el('textarea', {
      class: 'input input-sm', rows: 3,
      placeholder: '**正确说法**（强烈建议填写）：填了之后系统会记住，后续生成会参考'
    });
    var submit = el('button', { type: 'button', class: 'btn btn-sm btn-primary' }, [
      icon('check', 'ic-xs'), el('span', { text: '提交质疑' })
    ]);
    var cancel = el('button', { type: 'button', class: 'btn btn-sm' }, [
      el('span', { text: '取消' })
    ]);
    var panel = el('div', { class: 'fb-panel' }, [
      el('div', { class: 'fb-panel-hint' }, [
        el('strong', { text: '质疑与纠错' }),
        el('div', {
          class: 'fb-panel-sub',
          text: '填了「正确说法」的质疑会立即成为后续生成的记忆，同类错误会减少复发；'
              + '只点"没用"则会进入偏好训练集，用于日后离线微调。'
        })
      ]),
      category, quoted, comment, corrected,
      el('div', { class: 'fb-panel-actions' }, [submit, cancel])
    ]);
    bar.appendChild(panel);
    category.focus();

    cancel.addEventListener('click', function () { panel.remove(); });
    submit.addEventListener('click', function () {
      submit.disabled = true;
      submitFeedback({
        verdict: 'challenge',
        category: category.value,
        comment: comment.value.trim(),
        corrected_text: corrected.value.trim(),
        quoted_text: quoted.value.trim(),
        target_type: 'message',
        topic: ($('topicInput') && $('topicInput').value || '').trim(),
        run_id: (state.run && state.run.id) || '',
        session_id: state.activeSessionId || null
      }).then(function (res) {
        if (!res.ok) {
          submit.disabled = false;
          status.textContent = '提交失败：' + res.error;
          return;
        }
        panel.remove();
        status.textContent = (res.data && res.data.message) || '已记录';
        challengeBtn.classList.add('is-on');
        disableAll();
        pushTool('phase', '已提交质疑',
          '类别：' + (category.options[category.selectedIndex] || {}).text
          + (corrected.value.trim() ? ' · 已作为纠错记忆生效' : ' · 未填正确说法，仅进入训练集'));
        refreshLearningBadge();
      });
    });
  });

  return bar;
}

function submitFeedback(body) {
  return API.feedback(body);
}

/** 刷新「学习闭环」徽标：让用户看得见反馈确实被系统吸收了。 */
function refreshLearningBadge() {
  var chip = $('learningChip');
  if (!chip) return;
  API.feedbackSummary().then(function (res) {
    if (!res.ok || !res.data) return;
    var d = res.data;
    chip.hidden = false;
    chip.textContent = '学习闭环 · 反馈 ' + (d.total || 0)
      + ' · 生效记忆 ' + (d.active_memories || 0);
    chip.title = '赞 ' + (d.up || 0) + ' / 踩 ' + (d.down || 0)
      + ' / 质疑 ' + (d.challenge || 0)
      + '｜其中 ' + (d.active_memories || 0) + ' 条纠错已作为生成时的记忆生效';
  });
}

/* ========================================================================
 * §15. 论文写作（基于用户实验数据）
 * ======================================================================== */

function collectManuscriptBrief() {
  var get = function (id) { var n = $(id); return n ? n.value.trim() : ''; };
  return {
    title: get('msTitle'),
    goal: get('msGoal'),
    hypothesis: get('msHypothesis'),
    design: get('msDesign'),
    population: get('msPopulation'),
    intervention: get('msIntervention'),
    outcomes: get('msOutcomes'),
    statistics: get('msStatistics'),
    data: get('msData'),
    results: get('msResults'),
    limitations: get('msLimitations'),
    journal: get('msJournal')
  };
}

function generateManuscript() {
  var brief = collectManuscriptBrief();
  var missing = [];
  if (!brief.goal) missing.push('研究目标');
  if (!brief.design) missing.push('研究设计');
  if (!brief.population) missing.push('研究对象');
  if (!brief.outcomes) missing.push('结局指标');
  var status = $('msStatus');
  if (missing.length) {
    if (status) status.textContent = '请先填写：' + missing.join('、');
    toast('还缺必要要素：' + missing.join('、'), 'warn', 6000);
    return;
  }
  var btn = $('msDraftBtn');
  if (btn) btn.disabled = true;
  if (status) status.textContent = '正在生成初稿（逐节撰写，可能需要几分钟）…';

  var useLib = $('msUseLibrary');
  API.manuscriptDraft({
    brief: brief,
    session_id: state.activeSessionId || null,
    use_library: !useLib || useLib.checked,
    top_k: 12,
    save: true
  }).then(function (res) {
    if (btn) btn.disabled = false;
    if (!res.ok) {
      if (status) status.textContent = '生成失败：' + res.error;
      toast('生成失败：' + res.error, 'error', 8000);
      return;
    }
    renderManuscript(res.data || {});
  });
}

function renderManuscript(data) {
  var status = $('msStatus');
  var box = $('msDraft');
  var checksBox = $('msChecks');
  var draft = data.draft || '';
  if (status) {
    status.textContent = '已生成 ' + draft.length + ' 字'
      + (data.manuscript_id ? '（稿件 #' + data.manuscript_id + '）' : '');
  }
  if (box) {
    box.hidden = false;
    setMarkdown(box, draft || '（没有生成任何内容）');
  }
  var copyBtn = $('msCopyBtn');
  if (copyBtn) {
    copyBtn.hidden = false;
    copyBtn.onclick = function () {
      copyToClipboard(draft, '已复制论文全文。');
    };
  }
  if (checksBox) {
    checksBox.hidden = false;
    clear(checksBox);
    checksBox.appendChild(buildChecksView(data.checks || {}));
  }
  var errs = (data.checks && data.checks.generation_errors) || [];
  errs.forEach(function (m) { pushError('论文生成：' + m); });
  pushTool('phase', '论文初稿已生成',
    (data.checks && data.checks.verdict === 'pass'
      ? '数字溯源校验通过' : '数字溯源校验有问题，请核对')
    + ' · ' + draft.length + ' 字');
}

/** 渲染数字溯源校验结果 —— 这是"AI 写论文"能不能信的关键。 */
function buildChecksView(checks) {
  var frag = document.createDocumentFragment();
  var verdict = checks.verdict || 'unknown';
  var headline = {
    pass: '数字溯源校验通过',
    warn: '数字溯源校验：有少量数字无法溯源',
    fail: '数字溯源校验未通过',
    no_numbers: '正文中没有出现数字',
    unknown: '未做校验'
  }[verdict] || '校验结果';

  var tone = (verdict === 'pass' || verdict === 'no_numbers') ? 'ok' : 'warn';
  frag.appendChild(el('div', { class: 'ms-check-head ms-check-' + tone }, [
    icon(verdict === 'pass' ? 'check' : 'alert', 'ic-sm'),
    el('div', {}, [
      el('strong', { text: headline }),
      el('div', {
        class: 'ms-check-sub',
        text: '共检出 ' + (checks.total || 0) + ' 个数字，'
          + (checks.ok || 0) + ' 个能找到出处，'
          + (checks.unverified_count || 0) + ' 个无法溯源。'
      })
    ])
  ]));
  if (checks.note) {
    frag.appendChild(el('p', { class: 'form-help', text: checks.note }));
  }
  var extras = [];
  if (checks.used_literature) extras.push('已结合本地知识库文献');
  if (checks.used_memories) extras.push('已应用你此前的纠错记忆');
  if (extras.length) {
    frag.appendChild(el('p', { class: 'form-help', text: extras.join(' · ') }));
  }
  var bad = checks.unverified || [];
  if (bad.length) {
    var list = el('div', { class: 'ms-unverified' });
    bad.forEach(function (item) {
      list.appendChild(el('div', { class: 'ms-unverified-item' }, [
        el('code', { text: item.value }),
        el('span', { class: 'ms-unverified-sentence', text: '出现在：' + (item.sentence || '（无法定位）') })
      ]));
    });
    frag.appendChild(el('div', {}, [
      el('div', { class: 'ms-check-sub', text: '以下数字在你提供的数据与本地文献中都没有找到出处，请逐条核对：' }),
      list
    ]));
  }
  return frag;
}

/** 生成“思考中”占位气泡，返回可替换的节点。 */
function pushThinking(label) {
  hideWelcome();
  var box = $('messageStream');
  if (!box) return null;
  var body = el('div', { class: 'msg-body' }, [
    el('span', { class: 'thinking' }, [
      el('span', { class: 'thinking-dot' }), el('span', { class: 'thinking-dot' }), el('span', { class: 'thinking-dot' }),
      el('span', { text: label || '思考中…' })
    ])
  ]);
  var node = el('div', { class: 'msg msg-agent msg-thinking' }, [
    el('div', { class: 'msg-head' }, [
      el('span', { class: 'msg-role', text: 'MedScholar' }),
      el('span', { class: 'msg-time', text: clockNow() })
    ]),
    body
  ]);
  box.appendChild(node);
  scrollChatToEnd(true);
  return node;
}

/** 把一个“思考中”气泡替换为正式内容。 */
function resolveThinking(node, role, text, markdown) {
  if (!node || !node.parentNode) return pushMessage(role, text, null, { markdown: markdown });
  var body = node.querySelector('.msg-body');
  clear(body);
  if (markdown) body.appendChild(renderMarkdown(text));
  else body.textContent = text;
  node.className = 'msg msg-' + role;
  var head = node.querySelector('.msg-role');
  if (head) head.textContent = role === 'error' ? '错误' : 'MedScholar';
  scrollChatToEnd(true);
  return node;
}

/* ==========================================================================
 * §12. 工具调用面板
 * --------------------------------------------------------------------------
 * 汇聚 phase / status / search_result / critique / review / done / error
 * 等 Agent 步骤，按时间顺序展示，便于复盘一次运行的完整轨迹。
 * ======================================================================== */

function pushTool(kind, title, text, sources, rows) {
  var box = $('toolList');
  if (!box) return;
  var empty = box.querySelector('.empty-hint');
  if (empty) box.removeChild(empty);

  var main = el('div', { class: 'tool-main' }, [
    el('div', { class: 'tool-text' }, [
      el('strong', { text: title + (text ? '：' : '') }),
      text ? el('span', { text: text }) : null
    ])
  ]);

  // 单源结果行（每源：标签 / 条数 或 错误 / 耗时）
  if (sources && sources.length) {
    var rowBox = el('div', { class: 'tool-srcrow' });
    sources.forEach(function (s) {
      var ok = s.ok !== false;
      rowBox.appendChild(el('span', {
        class: 'src-chip ' + (s.skipped ? 'is-skip' : (ok ? 'is-ok' : 'is-bad')),
        title: s.error || ''
      }, [
        icon(ok ? 'check' : 'x', 'ic-xs'),
        el('span', { class: 'src-chip-name', text: s.label || sourceLabel(s.name) }),
        el('span', { class: 'src-chip-count', text: s.skipped ? '已跳过' : (ok ? (s.count || 0) + ' 条' : '✖ ' + (s.error || '失败')) }),
        s.duration_ms !== undefined ? el('span', { class: 'src-chip-ms', text: fmtMs(s.duration_ms) }) : null
      ]));
    });
    main.appendChild(rowBox);
  }

  // 通用明细行（label + text）
  if (rows && rows.length) {
    var sub = el('div', { class: 'tool-sub' });
    rows.forEach(function (r) {
      sub.appendChild(el('div', { class: 'tool-text' }, [
        el('span', { class: 'field-label', text: (r.label || '') + ' ' }),
        el('span', { text: r.text || '' })
      ]));
    });
    main.appendChild(sub);
  }

  box.appendChild(el('div', { class: 'tool-item', 'data-kind': kind }, [
    el('span', { class: 'tool-kind', text: kind }),
    main,
    el('span', { class: 'tool-time', text: clockNow() })
  ]));

  state.toolEntries++;
  var count = $('toolCount');
  if (count) count.textContent = String(state.toolEntries);
  box.scrollTop = box.scrollHeight;
}

function bindToolPanel() {
  var toggle = $('toolToggle');
  var body = $('toolBody');
  if (!toggle || !body) return;
  toggle.addEventListener('click', function () {
    var open = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
    body.hidden = open;
  });
}

/** 更新对话栏顶部的即时状态文字（不覆盖阶段徽标，避免两者互相刷掉）。 */
function setStatus(text) {
  var line = $('runStatusLine');
  if (!line) return;
  if (!text) { line.hidden = true; line.textContent = ''; return; }
  line.textContent = text;
  line.title = text;
  line.hidden = false;
}

function resetTools() {
  var box = $('toolList');
  if (box) {
    clear(box);
    box.appendChild(el('p', { class: 'empty-hint', text: '尚无工具调用。运行研究工作流后，这里会逐条显示每个 Agent 步骤。' }));
  }
  state.toolEntries = 0;
  var count = $('toolCount');
  if (count) count.textContent = '0';
}

/* ==========================================================================
 * §13. Agent 运行与 SSE
 * --------------------------------------------------------------------------
 * POST /api/agent/run       启动运行，取得 run_id
 * GET  /api/agent/stream/{id}  EventSource 订阅全部事件类型
 * POST /api/agent/approve/{id} 人工审批（approve / revise / cancel）
 * POST /api/agent/cancel/{id}  中止
 * ======================================================================== */

/** 启动一次研究工作流。topic 为空时取工具栏课题输入。 */
/**
 * 判断一段文字是否只是输入框的**占位提示**，而不是用户真正输入的课题。
 *
 * 为什么需要这个：实测出现过一次课题被写成
 * 「例如：加速rTMS治疗卒中后抑郁的疗效与安全性」—— 正是 `topicInput` 的
 * placeholder 原文。虽然代码里没有任何地方会把 placeholder 写进输入框
 * （已逐处排查：没有 value 属性、autocomplete=off、localStorage 存取正常），
 * 但浏览器 autofill 或粘贴都可能造成同样结果。
 *
 * 与其纠结它是怎么进来的，不如让"占位提示"在任何路径下都不可能被当成真课题 ——
 * 因为它一旦被当成课题，就会启动一次十几分钟、最终跑出无意义结果的研究流程。
 */
function isPlaceholderTopic(text) {
  var box = $('topicInput');
  var value = String(text || '').trim();
  if (!value) return true;
  if (!box) return false;

  var placeholder = (box.getAttribute('placeholder') || '').trim();
  if (!placeholder) return false;

  // 去掉「例如：」「示例:」这类引导词后再比一次，避免措辞微调导致漏判
  var strip = function (s) {
    return s.replace(/^(例如|示例|比如|举例|如)\s*[：:]\s*/, '').trim();
  };
  if (value === placeholder) return true;

  var a = strip(value);
  var b = strip(placeholder);
  if (!b) return false;
  return a === b || value === b;
}

function startAgentRun(topicOverride) {
  if (state.run && !state.run.finished) {
    toast('已有运行进行中，请先等待完成或点击「中止」。', 'warn');
    return;
  }
  var topic = (topicOverride !== undefined && topicOverride !== null ? topicOverride : $('topicInput').value) || '';
  topic = String(topic).trim();
  if (isPlaceholderTopic(topic)) {
    // 顺手清掉这个脏值，避免 localStorage 反复把它带回来
    var box = $('topicInput');
    if (box && box.value.trim() && isPlaceholderTopic(box.value)) box.value = '';
    lsSet(STORE_KEYS.topic, '');
    toast('请先输入研究课题（输入框里灰色显示的只是示例提示，不是已填内容）。', 'warn', 6000);
    if (box) box.focus();
    return;
  }
  $('topicInput').value = topic;
  lsSet(STORE_KEYS.topic, topic);

  var payload = {
    topic: topic,
    sources: selectedSources(),
    project_id: null,
    session_id: state.activeSessionId || null,
    require_approval: state.settings.requireApproval !== false,
    offline: state.settings.offline === true,
    review_min_chars: state.settings.reviewMin || 4000,
    review_max_chars: state.settings.reviewMax || 8000
  };

  setRunBusy(true);
  resetTools();
  setPhase('');
  clearPhaseWarn();
  setStatus('');
  pushMessage('user', topic);
  var thinking = pushThinking('正在规划研究方案…');
  pushTool('phase', '启动运行', '课题：' + topic + ' · 数据源：' + (payload.sources.length ? payload.sources.map(sourceLabel).join('、') : '未选择')
    + ' · 目标正文：' + payload.review_min_chars + '~' + payload.review_max_chars + ' 字');

  API.agentRun(payload).then(function (res) {
    if (!res.ok) {
      setRunBusy(false);
      resolveThinking(thinking, 'error', '启动研究失败：' + res.error);
      pushTool('error', '启动失败', res.error);
      return;
    }
    var d = res.data || {};
    if (!d.run_id) {
      setRunBusy(false);
      resolveThinking(thinking, 'error', '后端未返回 run_id，无法订阅事件流。请检查后端版本是否与 docs/API.md 一致。');
      return;
    }
    state.run = {
      id: d.run_id,
      sessionId: d.session_id,
      phase: '',
      finished: false,
      buffer: '',
      thinkingEl: thinking,
      startedAt: Date.now(),
      streamEl: null,
      renderPending: false
    };
    if (d.session_id) state.activeSessionId = d.session_id;
    setStatus('已启动运行 ' + String(d.run_id).slice(0, 8) + '，正在连接事件流…');
    pushTool('status', '运行已创建', 'run_id = ' + d.run_id + (d.session_id ? ' · session_id = ' + d.session_id : ''));
    openStream(d.run_id);
  });
}

/** 打开 SSE 事件流并注册全部事件类型。 */
function openStream(runId) {
  closeStream();
  var url = apiUrl('/api/agent/stream/' + encodeURIComponent(runId));
  var es;
  try {
    es = new window.EventSource(url);
  } catch (err) {
    pushError('无法建立事件流连接：' + (err && err.message ? err.message : '浏览器不支持 EventSource'));
    setRunBusy(false);
    return;
  }
  state.stream = es;

  SSE_TYPES.forEach(function (type) {
    es.addEventListener(type, function (ev) { handleSseEvent(type, ev); });
  });

  // 未命名事件（只有 data:）作为兜底：若 JSON 内带 type 字段则按类型分派。
  es.onmessage = function (ev) {
    var payload = safeParse(ev && ev.data);
    if (payload && typeof payload === 'object' && (payload.type || payload.event)) {
      handleSseEvent(String(payload.type || payload.event), { data: JSON.stringify(payload) });
    } else if (payload && payload.message) {
      handleSseEvent('status', { data: JSON.stringify(payload) });
    }
  };

  // 注意：EventSource 的连接层错误与后端自定义的 `event: error` 都派发为
  // type === 'error'，只能通过是否携带 data 字段区分。
  es.addEventListener('error', function (ev) {
    if (ev && typeof ev.data === 'string' && ev.data) {
      handleSseEvent('error', ev);
      return;
    }
    onStreamError();
  });

  es.addEventListener('open', function () {
    setStatus('事件流已连接。');
  });
}

function safeParse(text) {
  if (!text) return null;
  try { return JSON.parse(text); } catch (e) {
    dbg('SSE 非法 JSON', text);
    return null;
  }
}

function onStreamError() {
  if (!state.run || state.run.finished) return;
  setStatus('事件流中断，浏览器正在自动重连…');
  var chip = $('runStateChip');
  if (chip) { chip.textContent = '连接中断，重连中'; chip.className = 'chip chip-warn'; chip.hidden = false; }
}

function closeStream() {
  if (state.stream) {
    try { state.stream.close(); } catch (e) { /* 忽略 */ }
    state.stream = null;
  }
}

/** 统一的事件分派：单个事件解析失败不会影响后续事件。 */
function handleSseEvent(type, ev) {
  var data = safeParse(ev && ev.data);
  if (data === null) data = {};
  dbg('SSE', type, data);
  try {
    switch (type) {
      case 'phase':             onPhase(data); break;
      case 'status':            onStatus(data); break;
      case 'token':             onToken(data); break;
      case 'plan':              onPlan(data); break;
      case 'awaiting_approval': onAwaitingApproval(data); break;
      case 'search_result':     onSearchResult(data); break;
      case 'papers':            onPapers(data); break;
      case 'critique':          onCritique(data); break;
      case 'artifact':          onArtifact(data); break;
      case 'review':            onReview(data); break;
      case 'error':             onStreamErrorEvent(data); break;
      case 'done':              onDone(data); break;
      default:                  dbg('未知 SSE 事件类型', type, data);
    }
  } catch (err) {
    // 渲染异常绝不能中断事件流
    dbg('处理事件出错', type, err);
    pushError('处理服务端事件「' + type + '」时出错：' + (err && err.message ? err.message : String(err)));
  }
}

/* --- 各事件处理器 --- */

/** phase：阶段切换（plan / execute / reflect / synthesize / done）。 */
function onPhase(data) {
  var phase = String(data.phase || '');
  setPhase(phase);
  pushTool('phase', '阶段切换', (data.label || PHASE_LABELS[phase] || phase));
  if (state.run) state.run.phase = phase;
}

function setPhase(phase) {
  var track = $('phaseTrack');
  if (!track) return;
  var idx = PHASE_ORDER.indexOf(phase);
  Array.prototype.forEach.call(track.querySelectorAll('li'), function (li) {
    var p = li.getAttribute('data-phase');
    var pi = PHASE_ORDER.indexOf(p);
    li.classList.toggle('is-active', phase !== '' && p === phase);
    // 被标成告警的阶段（如规划降级）不再显示成绿色对勾，即使后面阶段已经推进
    li.classList.toggle(
      'is-done',
      idx >= 0 && pi >= 0 && pi < idx && !li.classList.contains('is-warn')
    );
  });
  var chip = $('runStateChip');
  if (chip) {
    if (phase) {
      chip.textContent = PHASE_LABELS[phase] || phase;
      chip.className = 'chip chip-accent';
      chip.hidden = false;
    } else {
      chip.hidden = true;
    }
  }
}

/** 把某个阶段标成「有问题地完成」——规划降级时不能显示成绿色对勾。 */
function markPhaseWarn(phase, hint) {
  var track = $('phaseTrack');
  if (!track) return;
  var li = track.querySelector('li[data-phase="' + phase + '"]');
  if (!li) return;
  li.classList.add('is-warn');
  li.classList.remove('is-done');
  if (hint) li.title = hint;
}

/** 重新开始一次运行时清掉上一次的告警标记。 */
function clearPhaseWarn() {
  var track = $('phaseTrack');
  if (!track) return;
  Array.prototype.forEach.call(track.querySelectorAll('li.is-warn'), function (li) {
    li.classList.remove('is-warn');
    li.removeAttribute('title');
  });
}

/** status：进度文字。 */
function onStatus(data) {
  var msg = data.message || '';
  if (!msg) return;
  setStatus(msg);
  pushTool('status', '进度', msg);
}

/** token：流式增量文本，累积到当前草稿与流式气泡中。 */
function onToken(data) {
  var text = data.text === undefined || data.text === null ? '' : String(data.text);
  if (!text) return;
  if (!state.run) return;
  state.run.buffer += text;

  // 首片 token 时把“思考中”占位替换为流式气泡
  if (!state.run.streamEl) {
    var node = state.run.thinkingEl && state.run.thinkingEl.parentNode
      ? state.run.thinkingEl
      : pushThinking('正在撰写…');
    if (node) {
      node.className = 'msg msg-agent msg-streaming';
      var body = node.querySelector('.msg-body');
      if (body) clear(body);
      var role = node.querySelector('.msg-role');
      if (role) role.textContent = 'MedScholar · 撰写中';
    }
    state.run.streamEl = node;
    var hint = node ? node.querySelector('.msg-body') : null;
    if (hint) hint.classList.add('stream-target');
  }

  scheduleStreamRender();
}

/** 以约 8 帧/秒的节流重绘流式 Markdown，避免每个 token 都重建 DOM。 */
function scheduleStreamRender() {
  if (!state.run || state.run.renderPending) return;
  state.run.renderPending = true;
  window.setTimeout(function () {
    if (!state.run) return;
    state.run.renderPending = false;
    var body = state.run.streamEl ? state.run.streamEl.querySelector('.msg-body') : null;
    if (!body) return;
    clear(body);
    body.appendChild(renderMarkdown(state.run.buffer));
    scrollChatToEnd(false);
  }, 120);
}

/** plan：规划完成，渲染结构化方案消息。 */
function onPlan(data) {
  var plan = data.plan || {};
  state.plan = plan;
  var node = pushMessage('agent', '', null, { markdown: false });
  var body = node.querySelector('.msg-body');
  clear(body);
  body.appendChild(buildPlanBlock(plan, false));
  scrollChatToEnd(true);
  if (plan.degraded) {
    // 规划其实降级了，阶段条不能显示成绿色对勾
    markPhaseWarn('plan', plan.degraded_reason || '规划降级为兜底方案');
    pushTool('phase', '规划降级', plan.degraded_reason || '已改用课题关键词 + 模板大纲');
  } else {
    pushTool('phase', '规划完成', planBlockSummary(plan));
  }
}

function planBlockSummary(plan) {
  var parts = [];
  if (plan.topic_zh) parts.push('中文课题：' + plan.topic_zh);
  if (plan.queries && plan.queries.length) parts.push(plan.queries.length + ' 条检索式');
  if (plan.outline && plan.outline.length) parts.push(plan.outline.length + ' 个章节');
  return parts.join(' · ') || '（方案内容为空）';
}

/** 构建方案展示块（审批面板复用同一渲染）。 */
function buildPlanBlock(plan, compact) {
  var frag = document.createDocumentFragment();

  // 兜底方案必须显眼地说清楚：否则用户会以为这是大模型定制的检索策略，
  // 批准后拿一个长中文串去检索，最后得到 0 篇文献却不知道为什么。
  if (plan && plan.degraded) {
    frag.appendChild(el('div', { class: 'notice notice-warn plan-degraded' }, [
      icon('alert'),
      el('div', { class: 'notice-body' }, [
        el('strong', { text: '这是兜底方案，不是按课题定制的检索策略。' }),
        el('div', { text: plan.degraded_reason || '大模型未能参与规划，已使用模板大纲与课题关键词检索。' }),
        el('div', { class: 'plan-degraded-hint', text: '建议先确认 Ollama 正在运行且模型可用（终端执行 ollama list），再重新发起研究；否则检索命中率会很低。' })
      ])
    ]));
  }

  var head = el('div', { class: 'plan-block' }, [
    el('div', { class: 'plan-h', text: '研究方案' })
  ]);
  if (plan.topic_zh) head.appendChild(el('p', { class: 'md-p', text: plan.topic_zh }));
  if (plan.topic_en) head.appendChild(el('p', { class: 'md-p', style: 'color:var(--c-text-2)', text: plan.topic_en }));
  frag.appendChild(head);

  // PICO
  var pico = plan.pico || {};
  var picoRows = [
    ['P 人群', pico.population],
    ['I 干预', pico.intervention],
    ['C 对照', pico.comparator],
    ['O 结局', Array.isArray(pico.outcomes) ? pico.outcomes.join('；') : pico.outcomes]
  ].filter(function (r) { return r[1]; });
  if (picoRows.length) {
    var dl = el('dl', { class: 'plan-kv' });
    picoRows.forEach(function (r) {
      dl.appendChild(el('dt', { text: r[0] }));
      dl.appendChild(el('dd', { text: String(r[1]) }));
    });
    frag.appendChild(el('div', { class: 'plan-block' }, [
      el('div', { class: 'plan-h', text: 'PICO 框架' }), dl
    ]));
  }

  // 检索式
  if (Array.isArray(plan.queries) && plan.queries.length) {
    var list = el('div', { class: 'plan-queries' });
    plan.queries.forEach(function (q) {
      var sources = Array.isArray(q.sources) ? q.sources : (q.sources ? [q.sources] : []);
      list.appendChild(el('div', { class: 'plan-query' }, [
        el('code', { class: 'plan-query-code', text: q.query || String(q) }),
        sources.length ? el('div', { class: 'plan-query-meta' }, sources.map(function (s) {
          return el('span', { class: 'chip', text: sourceLabel(s) });
        })) : null,
        q.rationale ? el('div', { class: 'plan-query-rationale', text: '理由：' + q.rationale }) : null
      ]));
    });
    frag.appendChild(el('div', { class: 'plan-block' }, [
      el('div', { class: 'plan-h', text: '检索式（' + plan.queries.length + ' 条）' }), list
    ]));
  }

  // MeSH / 年份
  var extras = [];
  if (Array.isArray(plan.mesh_terms) && plan.mesh_terms.length) {
    extras.push(el('div', { class: 'plan-block' }, [
      el('div', { class: 'plan-h', text: 'MeSH 主题词' }),
      el('div', { class: 'taglist' }, plan.mesh_terms.map(function (t) { return el('span', { class: 'tag', text: t }); }))
    ]));
  }
  if (plan.year_from) {
    extras.push(el('div', { class: 'plan-block' }, [
      el('div', { class: 'plan-h', text: '年份范围' }),
      el('p', { class: 'md-p', text: plan.year_from + ' 年至今' })
    ]));
  }
  extras.forEach(function (n) { frag.appendChild(n); });

  // 关键问题
  if (Array.isArray(plan.key_questions) && plan.key_questions.length) {
    var ul = el('ul', { class: 'md-ul' });
    plan.key_questions.forEach(function (q) { ul.appendChild(el('li', { text: String(q) })); });
    frag.appendChild(el('div', { class: 'plan-block' }, [
      el('div', { class: 'plan-h', text: '关键问题' }), ul
    ]));
  }

  // 大纲
  if (Array.isArray(plan.outline) && plan.outline.length) {
    var ol = el('div', { class: 'plan-outline' });
    plan.outline.forEach(function (sec, i) {
      var item = el('div', { class: 'plan-outline-item' }, [
        el('div', { class: 'plan-outline-title', text: (sec.title || ('第 ' + (i + 1) + ' 节')) })
      ]);
      if (Array.isArray(sec.points) && sec.points.length) {
        var pts = el('ul', { class: 'plan-outline-points' });
        sec.points.forEach(function (pt) { pts.appendChild(el('li', { text: String(pt) })); });
        item.appendChild(pts);
      }
      ol.appendChild(item);
    });
    frag.appendChild(el('div', { class: 'plan-block' }, [
      el('div', { class: 'plan-h', text: '大纲' }), ol
    ]));
  }

  if (!compact && !frag.childNodes.length) {
    frag.appendChild(noticeNode('warn', '规划数据为空或字段与契约不符，请检查后端返回的 plan 结构。'));
  }
  return frag;
}

/**
 * 结束一个已经不可能再提交的审批面板。
 *
 * 场景：服务被关掉/重启（运行随之中断）、运行已经自己跑完、或后端已经审批过。
 * 旧实现把面板原样留在页面上，按钮仍可点，点下去只得到
 * 「该运行不存在、已完成或已经审批过（可修改后重试）」——而"重试"根本没有用，
 * 因为那个运行已经不存在了。
 */
function resolveApproval(reason) {
  var panel = state.approval && state.approval.el;
  state.approval = null;
  if (!panel || !panel.parentNode) return false;
  if (panel.classList.contains('is-resolved') || panel.classList.contains('is-cancelled')) {
    return false;
  }
  panel.classList.add('is-resolved', 'is-stale');
  Array.prototype.forEach.call(panel.querySelectorAll('button'), function (b) { b.disabled = true; });
  var ta = panel.querySelector('textarea');
  if (ta) ta.disabled = true;
  var wait = panel.querySelector('.approval-wait');
  if (wait) wait.hidden = true;

  var h3 = panel.querySelector('.approval-head h3');
  if (h3) h3.textContent = '本次审批已失效';
  var foot = panel.querySelector('.approval-foot');
  if (foot) {
    clear(foot);
    foot.appendChild(el('span', { class: 'chip chip-warn', text: '运行已不在进行中' }));
  }
  panel.appendChild(el('div', { class: 'notice notice-warn' }, [
    icon('alert'),
    el('div', { class: 'notice-body', text: reason || '本次运行已经结束或中断，审批不再适用。请重新发起研究。' })
  ]));
  return true;
}

/** awaiting_approval：显示审批面板（批准 / 修改后继续 / 取消）。 */
function onAwaitingApproval(data) {
  var runId = data.run_id || (state.run && state.run.id);
  var plan = data.plan || state.plan || {};
  state.plan = plan;

  // 若后端未先发 plan 事件，这里补一条方案消息
  if (!document.querySelector('.plan-block')) onPlan({ plan: plan });

  pushTool('phase', '等待人工审批', '请确认研究方案后选择批准、修改或取消。');

  var feedback = el('textarea', {
    rows: 2,
    placeholder: '如需调整，请在此填写修改意见（点击「修改后继续」时提交给后端）'
  });

  var btnApprove = el('button', { type: 'button', class: 'btn btn-primary btn-sm', 'data-approval-action': 'approve' }, [icon('check', 'ic-xs'), el('span', { text: '批准执行' })]);
  var btnRevise  = el('button', { type: 'button', class: 'btn btn-sm', 'data-approval-action': 'revise' }, [icon('undo', 'ic-xs'), el('span', { text: '修改后继续' })]);
  var btnCancel  = el('button', { type: 'button', class: 'btn btn-cancel btn-sm', 'data-approval-action': 'cancel' }, [icon('x', 'ic-xs'), el('span', { text: '取消' })]);
  var wait = el('span', { class: 'approval-wait', text: '等待后端响应…', hidden: true });

  var foot = el('div', { class: 'approval-foot' }, [
    el('div', { class: 'approval-feedback' }, [feedback]),
    btnApprove, btnRevise, btnCancel, wait
  ]);

  var panel = el('section', { class: 'approval' }, [
    el('header', { class: 'approval-head' }, [
      icon('shield'),
      el('h3', { text: '需要人工审批' }),
      el('span', { class: 'chip chip-accent', text: 'run ' + String(runId || '').slice(0, 8) })
    ]),
    el('div', { class: 'approval-body' }, [buildPlanBlock(plan, false)]),
    foot
  ]);

  hideWelcome();
  var box = $('messageStream');
  if (box) {
    box.appendChild(panel);
    scrollChatToEnd(true);
  }

  state.approval = { runId: runId, el: panel, locked: false };

  function lock() {
    if (state.approval && state.approval.locked) return false;
    state.approval.locked = true;
    [btnApprove, btnRevise, btnCancel].forEach(function (b) { b.disabled = true; });
    feedback.disabled = true;
    wait.hidden = false;
    return true;
  }

  btnApprove.addEventListener('click', function () {
    if (!lock()) return;
    sendApproval(runId, 'approve', '');
  });
  btnRevise.addEventListener('click', function () {
    var text = (feedback.value || '').trim();
    if (!text) {
      toast('请先填写修改意见，再点击「修改后继续」。', 'warn');
      feedback.focus();
      return;
    }
    if (!lock()) return;
    sendApproval(runId, 'revise', text);
  });
  btnCancel.addEventListener('click', function () {
    if (!lock()) return;
    sendApproval(runId, 'cancel', '');
  });
  feedback.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      btnRevise.click();
    }
  });
}

/** 提交审批决定：POST /api/agent/approve/{run_id}。 */
function sendApproval(runId, decision, feedbackText) {
  if (!runId) {
    toast('缺少 run_id，无法提交审批。', 'error');
    return;
  }
  var body = { decision: decision };
  if (decision === 'revise' && feedbackText) body.feedback = feedbackText;

  API.agentApprove(runId, body).then(function (res) {
    var panel = state.approval && state.approval.el;
    var wait = panel ? panel.querySelector('.approval-wait') : null;
    if (wait) wait.hidden = true;

    if (!res.ok) {
      // 审批失败 → 解锁按钮，允许用户重试
      if (state.approval) state.approval.locked = false;
      var gone = /不存在|已完成|已经审批过/.test(String(res.error || ''));
      if (gone && resolveApproval(
        '这个运行在后端已经不存在或已结束（常见原因：服务被重启过、或运行已经自己跑完）。'
        + '审批按钮已经没有意义，请重新发起一次研究。'
      )) {
        toast('该运行已结束，审批不再适用。请重新发起研究。', 'warn');
        return;
      }
      if (panel) {
        Array.prototype.forEach.call(panel.querySelectorAll('button'), function (b) { b.disabled = false; });
        var ta = panel.querySelector('textarea');
        if (ta) ta.disabled = false;
        panel.classList.add('is-cancelled');
        panel.appendChild(el('div', { class: 'notice notice-error' }, [
          icon('alert'), el('div', { class: 'notice-body', text: '审批提交失败：' + res.error + '（可修改后重试）' })
        ]));
      }
      toast('审批提交失败：' + truncate(res.error, 120), 'error');
      return;
    }

    if (panel) {
      panel.classList.add(decision === 'cancel' ? 'is-cancelled' : 'is-resolved');
      var h3 = panel.querySelector('.approval-head h3');
      if (h3) h3.textContent = decision === 'approve' ? '已批准，继续执行'
        : (decision === 'revise' ? '已提交修改意见，按新方案继续' : '已取消本次运行');
      var foot = panel.querySelector('.approval-foot');
      if (foot) {
        clear(foot);
        foot.appendChild(el('span', { class: 'chip chip-ok', text: '决定：' + decision + ' · ' + clockNow() }));
        if (feedbackText) foot.appendChild(el('span', { class: 'chip', text: '修改意见：' + truncate(feedbackText, 80) }));
      }
    }
    pushMessage('user', decision === 'approve' ? '批准执行。'
      : (decision === 'revise' ? ('修改后继续：' + feedbackText) : '取消本次运行。'));
    pushTool('phase', '审批已提交', '决定：' + decision + (feedbackText ? ' · 意见：' + truncate(feedbackText, 60) : ''));

    if (decision === 'cancel') {
      setRunBusy(false);
      closeStream();
      if (state.run) state.run.finished = true;
      var chip = $('runStateChip');
      if (chip) { chip.textContent = '已取消'; chip.className = 'chip chip-quiet'; chip.hidden = false; }
      // 兜底：同时调用取消端点，避免后端仍处于等待状态
      if (runId) API.agentCancel(runId).then(function (r2) { if (!r2.ok) dbg('cancel fallback failed', r2.error); });
    } else {
      setStatus(decision === 'approve' ? '已批准，正在执行检索…' : '已提交修改意见，等待后端更新方案…');
    }
  });
}

/** search_result：单条检索式的结果，含各源条数与错误。 */
function onSearchResult(data) {
  var query = data.query || '(未命名检索式)';
  var sources = Array.isArray(data.sources) ? data.sources : [];
  var count = data.count === undefined ? (Array.isArray(data.items) ? data.items.length : 0) : data.count;
  var rows = [];
  if (data.saved) rows.push({ label: '入库', text: '新增 ' + (data.saved['new'] || 0) + ' · 更新 ' + (data.saved.updated || 0) });
  pushTool('search', '检索式结果', '“' + query + '” → ' + count + ' 条', sources, rows);

  if (Array.isArray(data.items) && data.items.length) upsertPapers(data.items, true);
  if (sources.length) renderSourceStatus(sources);

  var failed = sources.filter(function (s) { return s.ok === false && s.skipped !== true; });
  if (failed.length) {
    pushStatus('提示：' + failed.length + ' 个数据源本次未返回结果（' + failed.map(function (s) { return s.label || sourceLabel(s.name); }).join('、') + '），其余来源结果仍然有效。');
  }
}

/** papers：汇总后的文献卡片列表。 */
function onPapers(data) {
  var items = data.items || [];
  var count = data.count === undefined ? items.length : data.count;
  upsertPapers(items, true);
  pushTool('papers', '文献汇总', '共 ' + count + ' 篇进入候选池');
  pushStatus('已汇总 ' + count + ' 篇候选文献，右侧「文献卡片」可勾选查看。');
  var badge = $('tabBadgePapers');
  if (badge && items.length) { badge.textContent = String(state.papers.length); badge.hidden = false; }
}

/** critique：Critic 评估结果。 */
function onCritique(data) {
  var assessments = Array.isArray(data.assessments) ? data.assessments : [];
  var node = pushMessage('agent', '', null, { markdown: false });
  var body = node.querySelector('.msg-body');
  clear(body);
  body.appendChild(el('p', { class: 'md-p' }, [el('strong', { text: '证据评估（Critic）' })]));
  if (data.overall) body.appendChild(el('p', { class: 'md-p', text: String(data.overall) }));
  if (assessments.length) {
    var list = el('div', { class: 'critique-list' });
    assessments.forEach(function (a) {
      var verdict = String(a.verdict || a.level || a.rating || '').toLowerCase();
      var cls = verdict.indexOf('low') === 0 || verdict === 'weak' ? 'low'
        : (verdict.indexOf('high') === 0 || verdict === 'strong' ? 'high' : 'medium');
      list.appendChild(el('div', { class: 'critique-item', 'data-verdict': cls }, [
        el('div', { class: 'critique-title' }, [
          el('span', { text: a.title || a.paper_title || a.claim || ('评估项 ' + (list.childNodes.length + 1)) }),
          a.verdict ? el('span', { class: 'chip', text: '证据等级 ' + a.verdict }) : null,
          a.score !== undefined ? el('span', { class: 'chip chip-quiet', text: '评分 ' + a.score }) : null
        ]),
        (a.note || a.comment || a.reason) ? el('div', { class: 'critique-note', text: String(a.note || a.comment || a.reason) }) : null
      ]));
    });
    body.appendChild(list);
  } else if (!data.overall) {
    body.appendChild(el('p', { class: 'empty-hint', text: '（未返回具体评估条目）' }));
  }
  scrollChatToEnd(true);
  pushTool('critique', '证据评估', (assessments.length ? assessments.length + ' 项评估' : '') + (data.overall ? ' · ' + truncate(String(data.overall), 80) : ''));
}

/** artifact：生成产物（综述草稿）。 */
function onArtifact(data) {
  addArtifact({
    artifact_id: data.artifact_id !== undefined ? data.artifact_id : data.id,
    title: data.title || '综述草稿',
    content: data.content || '',
    fmt: data.fmt || 'markdown'
  });
  pushTool('artifact', '生成产物', (data.title || '综述草稿') + (data.fmt ? ' · ' + data.fmt : '') +
    (data.content ? ' · ' + data.content.length + ' 字' : ''));
  var node = pushMessage('agent', '', null, { markdown: false });
  var body = node.querySelector('.msg-body');
  clear(body);
  body.appendChild(el('p', { class: 'md-p' }, [
    el('strong', { text: '已生成产物：' + (data.title || '综述草稿') }),
    el('span', { text: '（正文见右侧「综述草稿」标签页）' })
  ]));
  switchTab('draft');
}

/** review：自我审查结果。 */
function onReview(data) {
  var issues = Array.isArray(data.issues) ? data.issues
    : (typeof data.issues === 'string' && data.issues ? [data.issues] : []);
  var node = pushMessage('agent', '', null, { markdown: false });
  var body = node.querySelector('.msg-body');
  clear(body);
  var verdict = data.verdict === undefined ? '' : String(data.verdict);
  body.appendChild(el('p', { class: 'md-p' }, [
    el('strong', { text: '自我审查：' }),
    el('span', { text: verdict || '（无结论）' }),
    data.score !== undefined ? el('span', { class: 'chip chip-quiet', text: '评分 ' + data.score }) : null
  ]));
  if (issues.length) {
    var ul = el('ul', { class: 'md-ul' });
    issues.forEach(function (it) {
      ul.appendChild(el('li', { text: typeof it === 'string' ? it : (it.message || it.issue || JSON.stringify(it)) }));
    });
    body.appendChild(ul);
  } else {
    body.appendChild(el('p', { class: 'md-p', style: 'color:var(--c-text-3)', text: '未发现需要修正的问题。' }));
  }
  scrollChatToEnd(true);
  pushTool('review', '自我审查', (verdict || '已完成') + (data.score !== undefined ? ' · 评分 ' + data.score : '') + (issues.length ? ' · ' + issues.length + ' 个问题' : ''));
}

/** error：非致命错误，随后流程会继续。 */
function onStreamErrorEvent(data) {
  var msg = data.message || data.detail || data.error;
  if (!msg) {
    // 后端自定义的 error 事件必定带 message；没有 message 说明这不是后端业务错误，
    // 而是连接层出了问题（服务被关掉、进程崩溃、端口变了）。
    // 早期版本在这里输出"服务端返回了一条未说明原因的错误"，对排查毫无帮助。
    var keys = data && typeof data === 'object' ? Object.keys(data).length : 0;
    msg = keys
      ? '服务端返回了无法识别的错误：' + JSON.stringify(data).slice(0, 200)
      : '与后端的事件流连接已断开：服务端没有返回任何错误内容。'
        + '请确认 MedScholar 的终端窗口仍然开着（关掉终端即等于关闭服务），'
        + '然后刷新页面重试。';
  }
  pushError(msg);
  pushTool('error', '运行告警', msg);
  setStatus('已记录一条错误，流程将继续（如已中断请查看上方错误详情）。');
  // 连接层断开时运行多半已经没了，审批面板留着只会让用户点出一个 409。
  if (!data.message && !data.detail && !data.error) {
    resolveApproval(
      '事件流已断开，这次运行很可能已经被中断（例如服务被关闭或重启）。'
      + '审批不再适用，请重新发起研究。'
    );
  }
}

/** done：运行结束。 */
function onDone(data) {
  var elapsed = data.elapsed_ms !== undefined ? data.elapsed_ms : (state.run ? Date.now() - state.run.startedAt : undefined);
  var usage = data.usage || {};
  var usageText = Object.keys(usage).map(function (k) { return k + '=' + usage[k]; }).join(' · ');

  setPhase('done');
  setRunBusy(false);
  closeStream();
  if (state.run) state.run.finished = true;
  // 运行已经结束，还留在页面上的审批面板必须失效（否则点下去只会得到 409）
  resolveApproval('本次运行已经结束，审批不再适用。如需调整方案请重新发起研究。');
  // 运行结束但可能没产出草稿（文献太少 / 模型失败 / 被中断）——
  // 让草稿页签给出准确原因，而不是一直显示"进入综合后会呈现"
  updateDraftEmptyHint();

  if (state.run && state.run.streamEl) {
    var streamNode = state.run.streamEl;
    streamNode.classList.remove('msg-streaming');
    var role = streamNode.querySelector('.msg-role');
    if (role) role.textContent = 'MedScholar · 撰写完成';
  }

  var summary = '研究流程结束，总耗时 ' + fmtMs(elapsed) + '。';
  if (state.papers.length) summary += ' 候选文献 ' + state.papers.length + ' 篇。';
  if (state.artifacts.length) summary += ' 生成产物 ' + state.artifacts.length + ' 份。';
  pushMessage('system', summary);
  pushTool('done', '运行完成', summary + (usageText ? ' · ' + usageText : ''));
  toast('研究流程已完成。', 'ok');

  var chip = $('runStateChip');
  if (chip) { chip.textContent = '已完成'; chip.className = 'chip chip-ok'; chip.hidden = false; }

  refreshLibrary();
  refreshSessions();
  refreshHealth({ silent: true });
}

/** 运行按钮/中止按钮的可用状态。 */
function setRunBusy(busy) {
  var runBtn = $('runAgentBtn');
  var cancelBtn = $('cancelRunBtn');
  if (runBtn) runBtn.disabled = busy;
  if (cancelBtn) cancelBtn.hidden = !busy;
  var hint = $('composerHint');
  if (hint) {
    hint.textContent = busy
      ? '运行进行中 · Enter 发送补充说明 · Ctrl/Cmd+Enter 启动研究'
      : 'Enter 发送 · Shift+Enter 换行 · Ctrl/Cmd+Enter 启动研究';
  }
}

/** 中止当前运行：先尝试审批端点取消，失败则退回取消端点。 */
function cancelRun() {
  var runId = state.run && state.run.id;
  if (!runId) {
    // 没有 run_id 时仍然尝试关闭事件流
    closeStream();
    setRunBusy(false);
    toast('已断开事件流。', 'info');
    return;
  }
  var btn = $('cancelRunBtn');
  if (btn) btn.disabled = true;
  if (state.approval && !state.approval.locked) {
    // 若正处于审批阶段，走审批取消路径（保持按钮状态一致）
    sendApproval(runId, 'cancel', '');
    if (btn) btn.disabled = false;
    return;
  }
  API.agentCancel(runId).then(function (res) {
    if (btn) btn.disabled = false;
    if (!res.ok) {
      toast('中止失败：' + truncate(res.error, 120), 'error');
      return;
    }
    closeStream();
    if (state.run) state.run.finished = true;
    setRunBusy(false);
    setStatus('运行已中止。');
    pushMessage('system', '运行已中止。');
    pushTool('done', '运行中止', '用户手动中止 run ' + String(runId).slice(0, 8));
  });
}

/* ==========================================================================
 * §14. 输入框、快捷键与设置对话框
 * ======================================================================== */

/** 自动增高输入框。 */
function autoGrow(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 180) + 'px';
}

/* --------------------------------------------------------------------------
 * 知识库问答（「提问」模式）
 *
 * 与「研究」模式的区别：只做「本地混合检索 → 拼接材料 → LLM 回答」，
 * 不联网检索、不写综述、不走审批，因此通常几十秒内就有答案。
 *
 * 后端 /api/ask 以 SSE 流式返回。这里用 fetch + ReadableStream 手工解析 ——
 * 因为 EventSource 不支持 POST，而问题需要放在请求体里。
 * ------------------------------------------------------------------------ */
var askState = { busy: false, buffer: '', el: null, renderPending: false, thinkingEl: null };

function handleAskBlock(block) {
  if (!block) return;
  var eventType = '';
  var dataLine = '';
  block.split('\n').forEach(function (line) {
    if (line.indexOf('event:') === 0) eventType = line.slice(6).trim();
    else if (line.indexOf('data:') === 0) dataLine += line.slice(5).trim();
  });
  if (!eventType || !dataLine) return;
  var data = safeParse(dataLine) || {};

  if (eventType === 'status') {
    if (askState.thinkingEl) {
      var b = askState.thinkingEl.querySelector('.msg-body');
      if (b) { clear(b); b.appendChild(el('span', { text: data.message || '处理中…' })); }
    }
  } else if (eventType === 'references') {
    pushTool('ask-retrieval', '知识库检索', data.message || ('命中 ' + (data.count || 0) + ' 篇'));
    askState.references = data.items || [];
  } else if (eventType === 'token') {
    onAskToken(data.text || '');
  } else if (eventType === 'error') {
    pushError(data.message || '提问失败。');
  } else if (eventType === 'done') {
    askState.finalAnswer = data.answer || askState.buffer;
  }
}

function onAskToken(text) {
  if (!text) return;
  askState.buffer += text;
  if (!askState.el) {
    var node = askState.thinkingEl && askState.thinkingEl.parentNode
      ? askState.thinkingEl
      : pushThinking('正在回答…');
    if (node) {
      node.className = 'msg msg-agent msg-streaming';
      var body = node.querySelector('.msg-body');
      if (body) clear(body);
      var role = node.querySelector('.msg-role');
      if (role) role.textContent = 'MedScholar · 回答中';
    }
    askState.el = node;
  }
  scheduleAskRender();
}

function scheduleAskRender() {
  if (askState.renderPending) return;
  askState.renderPending = true;
  window.setTimeout(function () {
    askState.renderPending = false;
    var body = askState.el ? askState.el.querySelector('.msg-body') : null;
    if (!body) return;
    clear(body);
    body.appendChild(renderMarkdown(askState.buffer));
    scrollChatToEnd(false);
  }, 120);
}

function askQuestion(question) {
  question = (question || '').trim();
  if (!question) return;
  if (askState.busy) { toast('上一个问题还在回答中，请稍候。', 'warn'); return; }

  askState = {
    busy: true, buffer: '', el: null, renderPending: false,
    thinkingEl: null, references: [], finalAnswer: ''
  };

  pushMessage('user', question);
  askState.thinkingEl = pushThinking('正在检索本地知识库…');
  setStatus('正在基于本地知识库回答问题…');
  setRunBusy(true);

  var topK = intOrNull($('askTopK') ? $('askTopK').value : null) || 8;
  var url = apiUrl('/api/ask');

  window.fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
    body: JSON.stringify({ question: question, top_k: topK })
  }).then(function (res) {
    if (!res.ok) throw new Error('HTTP ' + res.status);
    if (!res.body || typeof res.body.getReader !== 'function') {
      throw new Error('当前浏览器不支持流式读取');
    }
    var reader = res.body.getReader();
    var decoder = new window.TextDecoder('utf-8');
    var carry = '';
    function pump() {
      return reader.read().then(function (out) {
        if (out.done) {
          if (carry) handleAskBlock(carry);
          return;
        }
        carry += decoder.decode(out.value, { stream: true });
        var parts = carry.split('\n\n');
        carry = parts.pop();
        parts.forEach(handleAskBlock);
        return pump();
      });
    }
    return pump();
  }).catch(function (err) {
    var msg = (err && err.message) ? err.message : String(err);
    pushError(
      '提问失败：' + msg + '。请确认 MedScholar 服务仍在运行（终端窗口是否还开着）。'
    );
  }).then(function () {
    askState.busy = false;
    setRunBusy(false);
    if (askState.el) {
      askState.el.classList.remove('msg-streaming');
      var role = askState.el.querySelector('.msg-role');
      if (role) role.textContent = 'MedScholar · 回答';
    } else if (askState.thinkingEl && askState.thinkingEl.parentNode) {
      clear(askState.thinkingEl.parentNode ? askState.thinkingEl.querySelector('.msg-body') : null);
    }
    // 回答完成后附上引用来源，便于跳转核对
    if (askState.references && askState.references.length && askState.el) {
      pushAskReferences(askState.references);
    }
    if (!askState.buffer) {
      pushError('模型没有返回任何内容。若使用本地模型，可能是模型未加载或 max_tokens 太小。');
    }
    setStatus('回答完成。');
  });
}

/** 回答末尾列出引用的文献，方便回查。 */
function pushAskReferences(items) {
  var node = pushMessage('agent', '', null, { markdown: false });
  if (!node) return;
  var body = node.querySelector('.msg-body');
  if (!body) return;
  clear(body);
  body.appendChild(el('p', { class: 'md-p', style: 'color:var(--c-text-3)', text: '本次回答依据的文献：' }));
  var ul = el('ul', { class: 'md-ul' });
  items.forEach(function (it) {
    var label = '[' + it.index + '] ' + (it.title || '');
    if (it.pub_year) label += '（' + it.pub_year + '）';
    ul.appendChild(el('li', { text: truncate(label, 110) }));
  });
  body.appendChild(ul);
  scrollChatToEnd(true);
}

function bindComposer() {
  var form = $('composerForm');
  var input = $('composerInput');
  if (!form || !input) return;

  input.addEventListener('input', function () { autoGrow(input); });

  input.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Enter') return;
    if (ev.isComposing || ev.keyCode === 229) return; // 中文输入法组合中
    if (ev.shiftKey) return;                          // Shift+Enter 换行（默认行为）
    ev.preventDefault();
    // 无论当前是哪种模式，Ctrl/Cmd+Enter 都直接启动完整研究流程（保留旧习惯）
    if (ev.ctrlKey || ev.metaKey) { startAgentRun(); return; }
    form.dispatchEvent(new Event('submit', { cancelable: true }));
  });

  // 「提问 / 研究」模式切换
  Array.prototype.forEach.call(document.querySelectorAll('[data-composer-mode]'), function (btn) {
    btn.addEventListener('click', function () {
      setComposerMode(btn.getAttribute('data-composer-mode'));
    });
  });
  setComposerMode(state.composerMode || 'ask', true);

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var text = (input.value || '').trim();
    if (!text) return;
    input.value = '';
    autoGrow(input);

    // 审批等待中：把输入内容作为「修改后继续」的意见提交
    if (state.approval && !state.approval.locked && state.approval.el &&
        state.approval.el.parentNode && !state.approval.el.classList.contains('is-resolved') &&
        !state.approval.el.classList.contains('is-cancelled')) {
      var ta = state.approval.el.querySelector('textarea');
      if (ta) { ta.value = text; }
      var reviseBtn = state.approval.el.querySelector('[data-approval-action="revise"]');
      if (reviseBtn) { reviseBtn.click(); return; }
    }

    // 已有研究运行进行中：作为补充说明展示（后端未提供运行中对话端点）
    if (state.run && !state.run.finished) {
      pushMessage('user', text);
      pushStatus('已记录补充说明。当前后端契约（docs/API.md）未提供运行中的对话端点，如需变更方案请在审批面板提交修改意见。');
      return;
    }

    // 提问模式：走知识库问答，不启动完整研究流程
    if ((state.composerMode || 'ask') === 'ask') {
      askQuestion(text);
      return;
    }

    // 研究模式：以该文本作为课题启动一次新运行
    pushMessage('user', text);
    startAgentRun(text);
  });

  var sendBtn = $('composerSendBtn');
  if (sendBtn) {
    sendBtn.addEventListener('click', function () {
      form.dispatchEvent(new Event('submit', { cancelable: true }));
    });
  }
}

/**
 * 刷新「综述草稿」页签的空状态提示。
 *
 * 为什么不能只写一句"进入综合阶段后会在此呈现"：如果运行被中断（最常见的原因是
 * **服务在运行过程中重启**），用户会一直等一个永远不会到来的草稿，界面却毫无提示。
 * 实测踩到过这个坑，因此这里按真实情况给出不同说明。
 */
function updateDraftEmptyHint() {
  var hint = $('draftEmptyHint');
  if (!hint) return;
  if (state.artifacts && state.artifacts.length) return;

  var run = state.run;
  if (run && run.interrupted) {
    hint.textContent = '上次运行被中断，没有产生草稿。'
      + '最常见的原因是服务在运行过程中被关闭或重启（关掉终端窗口即等于关闭服务）。'
      + '请重新发起研究；已检索并入库的文献不会丢失。';
    hint.classList.add('empty-hint-warn');
    return;
  }
  if (run && !run.finished) {
    hint.textContent = '本次运行尚未进入「综合」阶段，草稿会在开始撰写后流式呈现在这里。';
    hint.classList.remove('empty-hint-warn');
    return;
  }
  if (run && run.finished) {
    hint.textContent = '本次运行已结束但没有产出草稿。'
      + '通常是因为检索到的文献太少、或模型调用失败 —— 请查看对话视口中的错误提示。'
      + (state.papers && state.papers.length ? '' : '（本次没有收集到文献，可在检索面板先联网检索一批文献。）');
    hint.classList.add('empty-hint-warn');
    return;
  }
  hint.textContent = '尚无综述草稿。Agent 进入「综合」阶段后会在此流式呈现。';
  hint.classList.remove('empty-hint-warn');
}

/** 从后端拉取运行历史，识别"上次运行被中断"这种情况。 */
function refreshRunHistory() {
  API.agentRuns().then(function (res) {
    if (!res.ok || !res.data) return;
    var runs = res.data.runs || [];
    var current = state.run;
    if (current && current.id) {
      var match = runs.filter(function (r) { return r.run_id === current.id; })[0];
      if (match && match.interrupted) {
        current.interrupted = true;
        pushStatus('注意：上次运行（' + current.id + '）在服务重启时被中断，'
          + '停止于「' + (match.phase_label || match.phase) + '」阶段，没有产出草稿。'
          + '已入库的文献不受影响，可重新发起研究。');
      }
    } else if (runs.length && runs[0].interrupted) {
      // 页面刷新后 state.run 为空，但后端有中断记录 → 明确告知
      pushStatus('检测到上次运行（' + runs[0].run_id + '）在服务重启时被中断，'
        + '停止于「' + (runs[0].phase_label || runs[0].phase) + '」阶段，没有产出草稿。'
        + '可重新发起研究。');
      state.run = state.run || { id: runs[0].run_id, interrupted: true, finished: true };
    }
    updateDraftEmptyHint();
  });
}

/**
 * 切换输入模式：`ask` = 知识库问答（默认），`research` = 完整研究流程。
 *
 * 之所以默认改成「提问」：多数时候用户只是想问一句（"这批文献讲了什么"、
 * "有没有关于剂量的证据"），而不是每次都跑十几分钟的综述流程。
 * 早期版本只能启动研究，导致一句普通提问被当成课题去检索写作。
 */
function setComposerMode(mode, silent) {
  state.composerMode = mode === 'research' ? 'research' : 'ask';
  lsSet(STORE_KEYS.composerMode, state.composerMode);

  Array.prototype.forEach.call(document.querySelectorAll('[data-composer-mode]'), function (btn) {
    var active = btn.getAttribute('data-composer-mode') === state.composerMode;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-composer-mode-opt]'), function (node) {
    node.hidden = node.getAttribute('data-composer-mode-opt') !== state.composerMode;
  });

  var isAsk = state.composerMode === 'ask';
  var input = $('composerInput');
  var hint = $('composerHint');
  var btnText = $('composerSendText');
  if (input) {
    input.placeholder = isAsk
      ? '问一个关于文献的问题…（Enter 提问，Shift+Enter 换行）'
      : '输入研究课题，启动检索 → 评估 → 综述流程…（Enter 启动，Shift+Enter 换行）';
  }
  if (hint) {
    hint.textContent = isAsk
      ? 'Enter 提问 · Shift+Enter 换行 · 切到「研究」可启动完整综述流程'
      : 'Enter 启动研究流程 · Shift+Enter 换行 · Ctrl/Cmd+Enter 也可启动';
  }
  if (btnText) btnText.textContent = isAsk ? '提问' : '开始研究';
  if (!silent) dbg('composer mode', state.composerMode);
}

function bindHotkeys() {
  document.addEventListener('keydown', function (ev) {
    var mod = ev.ctrlKey || ev.metaKey;
    if (!mod) {
      if (ev.key === 'Escape') closeSettings();
      return;
    }
    // Ctrl/Cmd+K：聚焦检索框（若面板折叠则展开）
    if (ev.key === 'k' || ev.key === 'K') {
      ev.preventDefault();
      var toggle = $('searchToggle');
      var body = $('searchBody');
      if (toggle && body && toggle.getAttribute('aria-expanded') !== 'true') {
        toggle.setAttribute('aria-expanded', 'true');
        body.hidden = false;
      }
      var q = $('searchQuery');
      if (q) { q.focus(); q.select(); }
      return;
    }
    // Ctrl/Cmd+Enter：启动研究工作流
    if (ev.key === 'Enter') {
      ev.preventDefault();
      startAgentRun();
    }
  });
}

/* --- 设置对话框 --- */
function openSettings(tab) {
  var modal = $('settingsModal');
  if (!modal) return;
  modal.hidden = false;
  setSettingsTab(tab || 'runtime');
  if (state.health) renderHealthDetail(state.health);
  else refreshHealth({ silent: true });
  var close = $('settingsCloseBtn');
  if (close) close.focus();
}

function closeSettings() {
  var modal = $('settingsModal');
  if (modal && !modal.hidden) modal.hidden = true;
}

function setSettingsTab(name) {
  Array.prototype.forEach.call(document.querySelectorAll('[data-mtab]'), function (t) {
    var active = t.getAttribute('data-mtab') === name;
    t.classList.toggle('is-active', active);
    t.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-mpane]'), function (p) {
    p.classList.toggle('is-active', p.getAttribute('data-mpane') === name);
  });
  // 切到「运行指标」时才去拉数据：指标每次都变，提前拉只会显示一份过期快照，
  // 而这个面板的用途恰恰是"我现在看到的数字是刚发生的"。
  if (name === 'metrics') refreshMetrics({});
}

/** 拉取运行指标（/api/metrics）。失败只提示，不影响其他面板。 */
function refreshMetrics(opts) {
  var box = $('metricsDetail');
  if (!box) return;
  if (!opts || !opts.silent) {
    clear(box);
    box.appendChild(el('p', { class: 'empty-hint', text: '加载中…' }));
  }
  api.metrics().then(function (data) {
    renderMetricsDetail(data);
  }).catch(function (err) {
    clear(box);
    box.appendChild(noticeNode('error', '运行指标获取失败：' + err.message));
  });
}

function renderMetricsDetail(m) {
  var box = $('metricsDetail');
  if (!box) return;
  clear(box);
  if (!m) { box.appendChild(noticeNode('error', '未取得运行指标。')); return; }
  var llm = m.llm || {};

  function fmtNum(n) {
    var v = Number(n || 0);
    return v.toLocaleString('zh-CN');
  }
  function fmtCost(yuan) {
    var v = Number(yuan || 0);
    if (!v) return '¥0（本地推理）';
    return '¥' + (v < 0.01 ? v.toFixed(4) : v.toFixed(2));
  }
  function card(title, main, sub) {
    return el('div', { class: 'metric-card' }, [
      el('div', { class: 'metric-title', text: title }),
      el('div', { class: 'metric-main', text: main }),
      el('div', { class: 'metric-sub', text: sub || '' })
    ]);
  }

  // ---- 第一排：成本与用量
  var cards = el('div', { class: 'metrics-grid' });
  cards.appendChild(card('LLM 调用', fmtNum(llm.calls),
    '成功 ' + fmtNum(llm.ok_calls) + ' · 失败 ' + fmtNum(llm.failed_calls)));
  cards.appendChild(card('Token 用量', fmtNum(llm.total_tokens),
    '输入 ' + fmtNum(llm.prompt_tokens) + ' · 输出 ' + fmtNum(llm.completion_tokens)));
  cards.appendChild(card('估算成本', fmtCost(llm.cost_yuan),
    '本地模型为 0；云端按量估算'));
  cards.appendChild(card('延迟 p50 / p95',
    Math.round(Number(llm.latency_ms_p50 || 0)) + ' / ' + Math.round(Number(llm.latency_ms_p95 || 0)) + ' ms',
    '慢在哪个阶段看下面'));
  box.appendChild(cards);

  // ---- 按阶段：回答"慢在哪、钱花在哪"
  var byPhase = llm.by_phase || {};
  var phaseKeys = Object.keys(byPhase);
  if (phaseKeys.length) {
    var phaseTbl = el('div', { class: 'metrics-table' });
    phaseTbl.appendChild(el('div', { class: 'form-label', text: '按阶段' }));
    phaseTbl.appendChild(el('div', { class: 'metrics-row metrics-row-head' }, [
      el('span', { text: '阶段' }), el('span', { text: '调用' }),
      el('span', { text: 'Token' }), el('span', { text: 'p95' })
    ]));
    phaseKeys.forEach(function (k) {
      var item = byPhase[k] || {};
      phaseTbl.appendChild(el('div', { class: 'metrics-row' }, [
        el('span', { text: k || '（未标注）' }),
        el('span', { text: fmtNum(item.calls) }),
        el('span', { text: fmtNum(Number(item.prompt_tokens || 0) + Number(item.completion_tokens || 0)) }),
        el('span', { text: Math.round(Number(item.latency_ms_p95 || 0)) + ' ms' })
      ]));
    });
    box.appendChild(phaseTbl);
  }

  // ---- 失败分类：回答"为什么失败"
  var byErr = llm.by_error_kind || {};
  var errKeys = Object.keys(byErr);
  if (errKeys.length) {
    var errBox = el('div', { class: 'metrics-table' });
    errBox.appendChild(el('div', { class: 'form-label', text: '失败分类（"失败 37 次"没有信息量，"429 占 30 次"才指向限流）' }));
    errKeys.forEach(function (k) {
      var item = byErr[k] || {};
      errBox.appendChild(el('div', { class: 'metrics-row metrics-row-2' }, [
        el('span', { class: 'chip chip-bad', text: k }),
        el('span', { text: fmtNum(item.calls) + ' 次' })
      ]));
    });
    box.appendChild(errBox);
  }

  // ---- 熔断器：后端是不是在连续失败
  var breakers = m.breakers || {};
  var breakerKeys = Object.keys(breakers);
  if (breakerKeys.length) {
    var brBox = el('div', { class: 'metrics-table' });
    brBox.appendChild(el('div', { class: 'form-label', text: '模型后端熔断状态' }));
    breakerKeys.forEach(function (k) {
      var b = breakers[k] || {};
      var state = String(b.state || 'closed');
      var cls = state === 'closed' ? 'chip chip-ok' : (state === 'open' ? 'chip chip-bad' : 'chip chip-warn');
      brBox.appendChild(el('div', { class: 'metrics-row metrics-row-2' }, [
        el('span', { class: cls, text: state === 'closed' ? '正常' : (state === 'open' ? '熔断中' : '半开探测') }),
        el('span', { text: k + ' · 连续失败 ' + fmtNum(b.failures) + (b.cooldown_remaining ? ' · 冷却 ' + Math.round(b.cooldown_remaining) + 's' : '') })
      ]));
    });
    box.appendChild(brBox);
  }

  // ---- 缓存命中率：这个缓存到底有没有用
  var caches = m.caches || {};
  var cacheKeys = Object.keys(caches);
  if (cacheKeys.length) {
    var cBox = el('div', { class: 'metrics-table' });
    cBox.appendChild(el('div', { class: 'form-label', text: '缓存命中率（命中率极低的缓存只是在消耗内存）' }));
    cacheKeys.forEach(function (k) {
      var c = caches[k] || {};
      var rate = Math.round(Number(c.hit_rate || 0) * 100);
      cBox.appendChild(el('div', { class: 'metrics-row metrics-row-2' }, [
        el('span', { text: k }),
        el('span', { text: rate + '% · 命中 ' + fmtNum(c.hits) + ' / 查询 ' + fmtNum(c.lookups) + ' · 占用 ' + fmtNum(c.size) })
      ]));
    });
    box.appendChild(cBox);
  }

  // ---- 注入扫描：语料有没有被投毒
  var inj = m.injection || {};
  var injBox = el('div', { class: 'metrics-table' });
  injBox.appendChild(el('div', { class: 'form-label', text: '检索内容注入扫描' }));
  var total = Number(inj.total_findings || 0);
  injBox.appendChild(el('div', { class: 'metrics-row metrics-row-2' }, [
    el('span', { class: total > 0 ? 'chip chip-warn' : 'chip chip-ok', text: total > 0 ? '发现可疑指令' : '未发现' }),
    el('span', {
      text: total > 0
        ? fmtNum(total) + ' 处 · 影响 ' + fmtNum(inj.blocks_with_findings) + ' 块材料（已按数据处理，不会执行）'
        : '所有检索材料均按"不可信数据"包裹后进入提示词'
    })
  ]));
  if (inj.last_finding) {
    injBox.appendChild(el('div', { class: 'metrics-row metrics-row-2' }, [
      el('span', { class: 'chip', text: inj.last_finding.kind || '未知' }),
      el('span', { text: (inj.last_finding.severity || '') + ' · ' + (inj.last_finding.excerpt || '') })
    ]));
  }
  box.appendChild(injBox);

  // ---- 最近调用：把"刚才发生了什么"摆出来
  var recent = m.llm_recent || [];
  if (recent.length) {
    var rBox = el('div', { class: 'metrics-table' });
    rBox.appendChild(el('div', { class: 'form-label', text: '最近调用（倒序）' }));
    recent.slice(-5).reverse().forEach(function (item) {
      rBox.appendChild(el('div', { class: 'metrics-row metrics-row-head-4' }, [
        el('span', { text: item.phase || '（未标注）' }),
        el('span', { text: (item.provider || '') + '/' + (item.model || '') }),
        el('span', { text: fmtNum(Number(item.prompt_tokens || 0) + Number(item.completion_tokens || 0)) + ' tok' }),
        el('span', {
          class: item.ok === false ? 'chip chip-bad' : 'chip chip-ok',
          text: item.ok === false ? ('失败 · ' + (item.error_kind || '')) : (Math.round(Number(item.latency_ms || 0)) + ' ms')
        })
      ]));
    });
    box.appendChild(rBox);
  }

  if (!llm.calls && !cacheKeys.length && !breakerKeys.length) {
    box.appendChild(noticeNode('warn', '还没有任何模型调用记录。发起一次研究或提问后，这里会出现 token、成本与延迟。'));
  }

  var note = $('metricsNote');
  if (note) {
    note.textContent = '指标来自本次服务进程（已运行 ' + Math.round(Number(m.uptime_s || 0)) +
      ' 秒），重启后清零；不包含任何文献内容或提示词正文。';
  }
}

function bindSettings() {
  var open = $('settingsBtn');
  if (open) open.addEventListener('click', function () { openSettings('runtime'); });
  var openHealth = $('healthBannerRetry');
  if (openHealth) openHealth.addEventListener('click', function () { refreshHealth({}); });
  var closeBtn = $('settingsCloseBtn');
  if (closeBtn) closeBtn.addEventListener('click', closeSettings);
  var backdrop = $('settingsModal');
  if (backdrop) {
    backdrop.addEventListener('click', function (ev) { if (ev.target === backdrop) closeSettings(); });
  }
  Array.prototype.forEach.call(document.querySelectorAll('[data-mtab]'), function (t) {
    t.addEventListener('click', function () { setSettingsTab(t.getAttribute('data-mtab')); });
  });
  var metricsBtn = $('metricsRefreshBtn');
  if (metricsBtn) metricsBtn.addEventListener('click', function () { refreshMetrics({}); });

  var apiBase = $('settingApiBase');
  var reqApproval = $('settingRequireApproval');
  var offline = $('settingOffline');
  var debug = $('settingDebug');
  var reviewMin = $('settingReviewMin');
  var reviewMax = $('settingReviewMax');
  if (apiBase) apiBase.value = state.settings.apiBase || '';
  if (reqApproval) reqApproval.checked = state.settings.requireApproval !== false;
  if (offline) offline.checked = state.settings.offline === true;
  if (debug) debug.checked = state.settings.debug === true;
  if (reviewMin) reviewMin.value = state.settings.reviewMin || 4000;
  if (reviewMax) reviewMax.value = state.settings.reviewMax || 8000;
  var libEnabled = $('settingLibraryEnabled');
  var libName = $('settingLibraryName');
  var libOpenUrl = $('settingLibraryOpenUrl');
  var libProxy = $('settingLibraryProxy');
  if (libEnabled) libEnabled.checked = state.settings.libraryEnabled === true;
  if (libName) libName.value = state.settings.libraryName || '';
  if (libOpenUrl) libOpenUrl.value = state.settings.libraryOpenUrl || '';
  if (libProxy) libProxy.value = state.settings.libraryProxyPrefix || '';

  var saveBtn = $('settingsSaveBtn');
  if (saveBtn) {
    saveBtn.addEventListener('click', function () {
      state.settings.apiBase = (apiBase && apiBase.value || '').trim().replace(/\/+$/, '');
      state.settings.requireApproval = !!(reqApproval && reqApproval.checked);
      state.settings.offline = !!(offline && offline.checked);
      state.settings.debug = !!(debug && debug.checked);
      var lo = toPositiveInt(reviewMin && reviewMin.value, 4000);
      var hi = toPositiveInt(reviewMax && reviewMax.value, 8000);
      if (lo > hi) { var t = lo; lo = hi; hi = t; }
      state.settings.reviewMin = Math.max(800, Math.min(lo, 40000));
      state.settings.reviewMax = Math.max(state.settings.reviewMin + 200, Math.min(hi, 40000));
      if (reviewMin) reviewMin.value = state.settings.reviewMin;
      if (reviewMax) reviewMax.value = state.settings.reviewMax;
      state.settings.libraryEnabled = !!(libEnabled && libEnabled.checked);
      state.settings.libraryName = ((libName && libName.value) || '图书馆').trim();
      state.settings.libraryOpenUrl = ((libOpenUrl && libOpenUrl.value) || '').trim();
      state.settings.libraryProxyPrefix = ((libProxy && libProxy.value) || '').trim();
      if (state.settings.libraryEnabled
          && !state.settings.libraryOpenUrl && !state.settings.libraryProxyPrefix) {
        toast('请至少填写「链接解析器地址」或「校外访问代理前缀」其中之一。', 'warn', 7000);
        return;
      }
      saveSettings();
      toast('设置已保存：综述目标 ' + state.settings.reviewMin + '~' + state.settings.reviewMax + ' 字，正在重新检测后端…', 'info');
      refreshHealth({}).then(function () { refreshLibrary(); refreshSessions(); });
      closeSettings();
    });
  }

  var resetLayout = $('settingResetLayout');
  if (resetLayout) {
    resetLayout.addEventListener('click', function () {
      var root = document.documentElement;
      root.style.setProperty('--left-w', '268px');
      root.style.setProperty('--mid-w', '560px');
      saveLayout();
      toast('已恢复默认栏宽。', 'ok');
    });
  }

  // 知识库维护
  var embedBtn = $('maintainEmbedBtn');
  var reindexBtn = $('maintainReindexBtn');
  if (embedBtn) {
    embedBtn.addEventListener('click', function () {
      var limit = intOrNull($('maintainLimit').value) || 200;
      maintain(embedBtn, '正在补齐向量（可能需要数分钟）…', function () {
        return API.embed({ limit: limit }).then(function (res) {
          if (!res.ok) return '补齐失败：' + res.error;
          var d = res.data || {};
          return '补齐完成：成功 ' + (d.embedded || 0) + ' · 跳过 ' + (d.skipped || 0) + ' · 失败 ' + (d.failed || 0);
        });
      });
    });
  }
  var fulltextBtn = $('maintainFulltextBtn');
  if (fulltextBtn) {
    fulltextBtn.addEventListener('click', function () {
      var limit = intOrNull($('maintainLimit').value) || 200;
      maintain(fulltextBtn, '正在为开放获取文献抓取全文（每篇约 1~5 秒，请耐心等待）…', function () {
        return API.fulltextBackfill({ limit: limit }).then(function (res) {
          if (!res.ok) return '补齐全文失败：' + res.error;
          var d = res.data || {};
          var lines = [];
          lines.push('补齐全文完成：成功 ' + (d.fetched || 0) + ' 篇 · 失败 ' + (d.failed || 0)
            + ' 篇 · 跳过非开放获取 ' + (d.skipped_not_oa || 0) + ' 篇');
          // 失败原因逐类列出 —— 只说"失败 N 条"用户无从判断，
          // 而实际上近一半是"文献本身就没有正文"这种正常情况。
          var reasons = d.reasons || [];
          if (reasons.length) {
            lines.push('失败原因明细：');
            reasons.forEach(function (r) {
              lines.push('  · ' + r.label + ' —— ' + r.count + ' 篇'
                + (r.permanent === false ? '（可重试）' : '（已记录，下次跳过）'));
            });
            if (d.permanent_failures) {
              lines.push('其中 ' + d.permanent_failures + ' 篇为永久性失败（文献本身无正文 / '
                + '出版商拒绝自动下载 / 非开放获取），已记录，下次「补齐全文」不会再重试它们。');
            }
          }
          return lines.join('\n');
        });
      });
    });
  }
  if (reindexBtn) {
    reindexBtn.addEventListener('click', function () {
      maintain(reindexBtn, '正在重建全文索引…', function () {
        return API.reindex().then(function (res) {
          return res.ok ? '索引重建完成。' : '重建失败：' + res.error;
        });
      });
    });
  }
}

/** 维护操作的通用执行/日志/刷新逻辑。 */
function maintain(btn, busyText, fn) {
  var log = $('maintainLog');
  if (btn) btn.disabled = true;
  if (log) {
    log.hidden = false;
    appendLog(log, busyText);
  }
  fn().then(function (msg) {
    if (btn) btn.disabled = false;
    if (log) appendLog(log, msg);
    toast(msg, /失败/.test(msg) ? 'error' : 'ok');
    refreshHealth({ silent: true }).then(refreshLibrary);
  }).catch(function (err) {
    if (btn) btn.disabled = false;
    if (log) appendLog(log, '执行异常：' + (err && err.message ? err.message : String(err)));
  });
}

function appendLog(log, text) {
  log.appendChild(document.createTextNode('[' + clockNow() + '] ' + text + '\n'));
  log.scrollTop = log.scrollHeight;
}

/* --- 顶部工具栏绑定 --- */
function bindToolbar() {
  // 课题输入持久化
  var topic = $('topicInput');
  var saved = lsGet(STORE_KEYS.topic, '');
  // 恢复上次课题，但**不恢复占位提示或空值** —— 否则一次脏写入会被永久带回来
  if (topic && typeof saved === 'string' && saved.trim() && !isPlaceholderTopic(saved)) {
    topic.value = saved;
  } else if (saved) {
    lsSet(STORE_KEYS.topic, '');
  }
  if (topic) {
    topic.addEventListener('input', function () {
      // 空值不写入：localStorage 里留个空串没有意义，还会掩盖真正的上次输入
      lsSet(STORE_KEYS.topic, topic.value.trim() ? topic.value : '');
    });
    topic.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' && !ev.isComposing && ev.keyCode !== 229) {
        ev.preventDefault();
        startAgentRun();
      }
    });
  }

  var runBtn = $('runAgentBtn');
  if (runBtn) runBtn.addEventListener('click', function () { startAgentRun(); });
  var cancelBtn = $('cancelRunBtn');
  if (cancelBtn) cancelBtn.addEventListener('click', cancelRun);

  // 引用格式
  var styleSel = $('citeStyleSelect');
  if (styleSel) {
    styleSel.addEventListener('change', function () {
      state.citeStyle = styleSel.value;
      lsSet(STORE_KEYS.citeStyle, state.citeStyle);
      updateCiteStyleLabel();
      if (state.citations.length) generateCitations();
    });
  }

  // 数据源下拉开合
  var srcBtn = $('sourceSelectBtn');
  var srcMenu = $('sourceMenu');
  if (srcBtn && srcMenu) {
    srcBtn.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var open = srcBtn.getAttribute('aria-expanded') === 'true';
      srcBtn.setAttribute('aria-expanded', open ? 'false' : 'true');
      srcMenu.hidden = open;
    });
    document.addEventListener('click', function (ev) {
      if (srcMenu.hidden) return;
      if (srcMenu.contains(ev.target) || srcBtn.contains(ev.target)) return;
      srcMenu.hidden = true;
      srcBtn.setAttribute('aria-expanded', 'false');
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll('[data-sources-all]'), function (btn) {
    btn.addEventListener('click', function () {
      var mode = btn.getAttribute('data-sources-all');
      state.sources = mode === 'none' ? [] : DEFAULT_SOURCES.slice();
      lsSet(STORE_KEYS.sources, state.sources);
      renderSourceMenu();
    });
  });

  // 导入题录文件
  var importBtn = $('importBtn');
  var importFile = $('importFile');
  if (importBtn && importFile) {
    importBtn.addEventListener('click', function () { importFile.click(); });
    importFile.addEventListener('change', function () {
      importFiles(importFile.files);
      importFile.value = '';  // 允许连续导入同一个文件
    });
  }

  // 论文写作
  var msBtn = $('msDraftBtn');
  if (msBtn) msBtn.addEventListener('click', generateManuscript);

  // 学习闭环徽标
  refreshLearningBadge();

  // 健康横幅关闭
  var closeBanner = $('healthBannerClose');
  if (closeBanner) {
    closeBanner.addEventListener('click', function () {
      state.bannerDismissed = true;
      renderBanner([]);
    });
  }

  // 左栏刷新
  var refreshBtn = $('refreshSessionsBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', function () { refreshSessions(); });
  var refreshKb = $('refreshKbBtn');
  if (refreshKb) refreshKb.addEventListener('click', function () { refreshKbStats(); });

  // 文献库过滤/排序
  var filter = $('libraryFilter');
  if (filter) {
    var t = null;
    filter.addEventListener('input', function () {
      if (t) window.clearTimeout(t);
      t = window.setTimeout(renderLibraryList, 140);
    });
  }
  var sortSel = $('librarySort');
  if (sortSel) sortSel.addEventListener('change', function () { refreshLibrary(); });

  // 右栏批量操作
  var selectAll = $('selectAllPapers');
  if (selectAll) {
    selectAll.addEventListener('change', function () {
      if (selectAll.checked) {
        state.papers.forEach(function (p) { state.selected[paperKey(p)] = true; });
      } else {
        state.selected = {};
      }
      renderPapers();
      renderLibraryList();
    });
  }
  var genBtn = $('generateCiteBtn');
  if (genBtn) genBtn.addEventListener('click', generateCitations);
  var expBtn = $('exportBtn');
  if (expBtn) expBtn.addEventListener('click', exportSelection);

  var copyCite = $('copyCitationsBtn');
  if (copyCite) {
    copyCite.addEventListener('click', function () {
      var text = state.citations.map(function (c) { return c.text; }).join('\n\n');
      if (!text) { toast('还没有可复制的引用，请先生成引用。', 'warn'); return; }
      copyToClipboard(text, '引用已复制到剪贴板。');
    });
  }
  var copyDraft = $('copyDraftBtn');
  if (copyDraft) {
    copyDraft.addEventListener('click', function () {
      var a = state.artifacts[state.activeArtifact];
      if (!a || !a.content) { toast('草稿内容为空。', 'warn'); return; }
      copyToClipboard(a.content, '草稿已复制到剪贴板。');
    });
  }
  var picker = $('draftPicker');
  if (picker) {
    picker.addEventListener('change', function () { showArtifact(parseInt(picker.value, 10) || 0); });
  }

  var clearChat = $('clearChatBtn');
  if (clearChat) {
    clearChat.addEventListener('click', function () {
      var box = $('messageStream');
      if (!box) return;
      clear(box);
      box.appendChild(el('p', { class: 'empty-hint', text: '对话视口已清空。' }));
      resetTools();
    });
  }

  // 引用标记点击（事件委托，覆盖所有渲染出来的 markdown）
  document.addEventListener('click', function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('.cite-mark-btn') : null;
    if (!btn) return;
    ev.preventDefault();
    var idx = parseInt(btn.getAttribute('data-cite-index'), 10);
    if (isFinite(idx)) highlightCitation(idx);
  });
}

/** 复制到剪贴板（含降级路径）。 */
function copyToClipboard(text, okMessage) {
  function fallback() {
    var ta = el('textarea', { style: 'position:fixed;top:-1000px;left:-1000px' });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    toast(ok ? okMessage : '复制失败，请手动选择文本复制。', ok ? 'ok' : 'warn');
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function () { toast(okMessage, 'ok'); }, fallback);
  } else {
    fallback();
  }
}

/* ==========================================================================
 * §15. 启动
 * ======================================================================== */

/* --------------------------------------------------------------------------
 * 首屏连接握手 + 断线自愈
 *
 * 为什么需要它：服务刚启动时 uvicorn 需要 2~4 秒才真正监听端口，
 * 而浏览器可能已经打开了页面。此时首屏的每个请求都会 "Failed to fetch"。
 *
 * 更关键的是**服务重启后页面要能自愈**：早期实现只重试约 25 秒就永久放弃，
 * 用户关掉服务再打开、或先开页面后开服务，页面就永远停在错误状态，
 * 必须手动刷新 —— 这正是反复收到"连不上后端"反馈的原因。
 * 现在改为：连不上就按 1.5s→5s 的间隔**一直重试**，连上后自动恢复。
 * ------------------------------------------------------------------------ */
function waitForBackend(attempt) {
  attempt = attempt || 1;
  var maxAttempts = 12;
  return API.ping().then(function (res) {
    if (res.ok) {
      if (attempt > 1) dbg('backend ready after attempt', attempt);
      return true;
    }
    if (attempt >= maxAttempts) return false;
    var delay = Math.min(400 * Math.pow(1.5, attempt - 1), 3000);
    setBootStatus('正在连接后端服务…（第 ' + attempt + ' 次尝试）');
    return new Promise(function (resolve) {
      window.setTimeout(function () {
        resolve(waitForBackend(attempt + 1));
      }, delay);
    });
  });
}

/** 首屏连接期间在欢迎区显示状态，避免用户以为卡死。 */
function setBootStatus(text) {
  var note = $('welcomeHealthNote');
  if (note) note.textContent = text;
}

/** 连不上时的持续重连：每 6 秒探一次，连上就自动把界面恢复。 */
function startReconnectLoop() {
  if (state.reconnectTimer) return;
  setBootStatus('尚未连接到后端，正在每 6 秒自动重试…（服务启动后会自动恢复，无需刷新）');
  state.reconnectTimer = window.setInterval(function () {
    API.ping().then(function (res) {
      if (!res.ok) return;
      window.clearInterval(state.reconnectTimer);
      state.reconnectTimer = null;
      dbg('reconnected');
      toast('已连接到后端服务。', 'ok');
      loadInitialData();
    });
  }, 6000);
}

/**
 * 后端把模型探测放在后台异步做，首屏拿到的 llm.ok / embedding.ok 可能是 null
 * （表示"检测中"）。这里持续轮询直到探测出结果。
 *
 * 为什么不能只等一次：模型不在内存时需要重新载入，实测 qwen3:8b 冷加载要 20~35 秒，
 * 只等 6 秒的话用户会一直停在"检测中"，还以为卡住了。
 * 最长轮询 90 秒，之后放弃（用户仍可点「重新检测」手动触发）。
 */
function pollHealthUntilProbed(attempt) {
  attempt = attempt || 1;
  var maxAttempts = 18;          // 18 × 5s = 90s
  if (attempt > maxAttempts) return;
  window.setTimeout(function () {
    refreshHealth({ silent: true }).then(function (h) {
      if (!h) return;
      var llmPending = h.llm && h.llm.ok === null;
      var embedPending = h.embedding && h.embedding.ok === null;
      if (llmPending || embedPending) pollHealthUntilProbed(attempt + 1);
    });
  }, 5000);
}

function stopReconnectLoop() {
  if (state.reconnectTimer) {
    window.clearInterval(state.reconnectTimer);
    state.reconnectTimer = null;
  }
}

function boot() {
  loadSettings();
  loadLayout();

  // 静态绑定
  bindToolbar();
  bindSearchPanel();
  bindComposer();
  bindHotkeys();
  bindToolPanel();
  bindTabs();
  bindSettings();
  bindGutter($('gutterLeft'), 'left');
  bindGutter($('gutterRight'), 'right');

  renderStyleSelect();
  updateSourceButton();

  setBootStatus('正在连接后端服务…');

  waitForBackend().then(function (ready) {
    if (!ready) {
      // 首轮重试耗尽：渲染错误状态，同时**继续在后台重连**，服务起来后自动恢复
      refreshHealth({});
      startReconnectLoop();
      return;
    }
    stopReconnectLoop();
    loadInitialData();
  });
}

/** 拉取首屏数据（连接建立后调用，也可在重连成功后再次调用）。 */
function loadInitialData() {
  refreshHealth({}).then(function (h) {
    // 后端异步探测模型，首屏可能拿到"检测中"；持续轮询直到出结果
    if (h && ((h.llm && h.llm.ok === null) || (h.embedding && h.embedding.ok === null))) {
      pollHealthUntilProbed(1);
    }
  });
  refreshLibrary();
  refreshSessions();
  refreshCiteStyles();
  // 拉一次运行历史：识别"上次运行被服务重启中断"这种情况，
  // 否则用户会对着一个永远等不到草稿的页面发呆
  refreshRunHistory();
  // 恢复现场：上次已经写好的综述 + 可以接着跑的阶段
  restoreLatestRun();
  API.config().then(function (res) {
    if (res.ok && res.data) {
      state.config = res.data;
      dbg('config', res.data);
    }
  });
}

/**
 * 页面加载时恢复现场。
 *
 * 之前的行为：综述明明已经写入数据库，刷新页面后「综述草稿」却是空的，
 * 只能看到一句"本次运行没有产出草稿"——用户以为白跑了，只能重新跑一遍。
 * 这里把已完成的产物和可续跑的阶段都取回来。
 */
function restoreLatestRun() {
  API.agentLatest().then(function (res) {
    if (!res.ok || !res.data || !res.data.run) return;
    var run = res.data.run;
    var steps = res.data.steps || [];
    state.latestRun = run;

    // 1) 已生成的综述直接显示出来，不必点会话
    if (run.artifact_id) {
      loadArtifactIntoBar(run.artifact_id, run.artifact_title);
    }

    // 2) 可以续跑的话给一条明确的提示
    if (run.resumable) {
      showResumeBanner(run, steps);
    } else if (steps.length) {
      var done = steps[steps.length - 1].phase;
      pushTool('phase', '已恢复上次的成果',
        '上次运行（' + String(run.run_id).slice(0, 8) + '）已完成「' + done + '」阶段，'
        + '其内容已从数据库恢复；已入库文献 ' + (run.papers || 0) + ' 篇。');
    }
    updateDraftEmptyHint();
  });
}

/** 把某份产物读进产物条并展示（刷新页面后也能看到上次的综述）。 */
function loadArtifactIntoBar(artifactId, title) {
  API.artifact(artifactId).then(function (res) {
    if (!res.ok || !res.data) return;
    var a = res.data.artifact || res.data;
    if (!a || a.id === undefined) return;
    var known = false;
    for (var i = 0; i < state.artifacts.length; i++) {
      if (String(state.artifacts[i].id) === String(a.id)) { known = true; break; }
    }
    if (!known) {
      state.artifacts.push({
        id: a.id,
        title: a.title || title || '综述草稿',
        fmt: a.fmt || 'markdown',
        content: a.content || '',
        char_count: a.char_count || (a.content ? a.content.length : 0)
      });
    }
    renderArtifactPicker();
    if (state.artifacts.length && state.activeArtifact === undefined) {
      showArtifact(state.artifacts.length - 1);
    }
  });
}

/** 在对话视口里放一张「从中断处继续」的卡片。 */
function showResumeBanner(run, steps) {
  var last = steps.length ? steps[steps.length - 1] : null;
  var at = last ? last.phase : (run.phase || '');
  var LABELS = { plan: '规划', execute: '执行', reflect: '反思', synthesize: '综合', review: '审查' };
  var box = $('messageStream');
  if (!box) return;

  var kept = [];
  if (run.papers) kept.push('文献 ' + run.papers + ' 篇');
  if (last && last.draft_chars) kept.push('草稿 ' + last.draft_chars + ' 字');
  if (last && last.has_critique) kept.push('文献评估');

  var btn = el('button', { type: 'button', class: 'btn btn-primary btn-sm' }, [
    icon('play', 'ic-xs'), el('span', { text: '从中断处继续' })
  ]);
  var wait = el('span', { class: 'approval-wait', text: '正在恢复…', hidden: true });

  var panel = el('section', { class: 'approval resume-card' }, [
    el('header', { class: 'approval-head' }, [
      icon('undo'),
      el('h3', { text: '上次的运行可以接着跑' }),
      el('span', { class: 'chip chip-warn', text: 'run ' + String(run.run_id).slice(0, 8) })
    ]),
    el('div', { class: 'approval-body' }, [
      el('p', { class: 'md-p', text: '课题：' + (run.topic || '') }),
      el('p', {
        class: 'md-p',
        text: '已完成到「' + (LABELS[at] || at || '未知') + '」阶段'
          + (kept.length ? '，可直接复用：' + kept.join('、') : '')
          + '。继续时会跳过已完成的阶段，不会重新检索、也不会重新撰写。'
      }),
      el('div', { class: 'approval-foot' }, [btn, wait])
    ])
  ]);

  box.appendChild(panel);
  scrollChatToEnd(true);

  btn.addEventListener('click', function () {
    btn.disabled = true;
    wait.hidden = false;
    resumeRun(run.run_id, panel);
  });
}

/** 调后端从断点继续，并接上事件流。 */
function resumeRun(runId, panel) {
  API.agentResume(runId).then(function (res) {
    if (!res.ok) {
      if (panel) panel.classList.add('is-stale');
      toast('继续失败：' + res.error, 'error');
      pushError('继续上次运行失败：' + res.error);
      return;
    }
    var d = res.data || {};
    setRunBusy(true);
    resetTools();
    setPhase('');
    clearPhaseWarn();
    state.run = {
      id: d.run_id || runId,
      sessionId: d.session_id,
      phase: '',
      finished: false,
      buffer: '',
      thinkingEl: null,
      startedAt: Date.now(),
      streamEl: null,
      renderPending: false,
      resumed: true
    };
    pushMessage('user', '从中断处继续。');
    pushTool('status', '继续运行',
      'run_id = ' + state.run.id + ' · 复用已完成阶段'
      + (typeof d.papers === 'number' ? ' · 复用文献 ' + d.papers + ' 篇' : ''));
    setStatus('已恢复运行，正在连接事件流…');
    openStream(state.run.id);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

})();
