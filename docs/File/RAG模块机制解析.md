# RAG 模块机制深度解析

> 覆盖混合RAG知识增强模块的完整链路：文档摄入（ETL）→ 向量化 → 混合检索 → 重排序 → 答案生成，含完整源码引用。

---

## 目录

- [一、模块全景](#一模块全景)
- [二、ETL 文档处理管线](#二etl-文档处理管线)
- [三、向量化与嵌入模型](#三向量化与嵌入模型)
- [四、存储引擎（Milvus 主 + 内存降级）](#四存储引擎milvus-主--内存降级)
- [五、检索系统（三模式 + RRF 融合）](#五检索系统三模式--rrf-融合)
- [六、重排序（Cross-Encoder）](#六重排序cross-encoder)
- [七、答案生成与引用标注](#七答案生成与引用标注)
- [八、RAGService 顶层服务](#八ragservice-顶层服务)
- [九、API 路由层](#九api-路由层)
- [十、完整请求链路](#十完整请求链路)
- [十一、关键设计决策总结](#十一关键设计决策总结)

---

## 一、模块全景

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  API 层 (app/api/routes/chat.py)                        │
│  /chat/rag | /rag/ingest | /rag/status | /rag/documents │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  RAGService (app/core/rag_service.py)                   │
│  顶层协调：ingest → embed → store → search → generate   │
└──┬──────────┬──────────┬──────────┬─────────────────────┘
   │          │          │          │
┌──▼──┐ ┌────▼────┐ ┌───▼───┐ ┌───▼──────────┐
│ETL   │ │Embedding│ │Store  │ │RAGGenerator   │
│Parser│ │Model    │ │Milvus │ │(答案生成+引用) │
│Chunk │ │         │ │内存降级│ │               │
└──────┘ └─────────┘ └──┬──┬─┘ └───────────────┘
                        │  │
                   ┌────▼──▼────┐
                   │ 检索三模式  │
                   │vector/     │
                   │keyword/    │
                   │hybrid(RRF) │
                   └────┬───────┘
                        │
                   ┌────▼────┐
                   │Reranker │
                   │(可选)    │
                   └─────────┘
```

### 1.2 代码文件分布

| 文件 | 职责 |
|------|------|
| [app/etl/parser.py](../app/etl/parser.py) | 文档解析（PDF/TXT → 纯文本） |
| [app/etl/chunker.py](../app/etl/chunker.py) | 文本分块（4 种策略，tiktoken 精确计数） |
| [app/etl/pipeline.py](../app/etl/pipeline.py) | ETL 流水线编排（parse → chunk → callback） |
| [app/core/rag_service.py](../app/core/rag_service.py) | RAG 顶层服务（底层实现：BM25 + Milvus Lite + RRF） |
| [app/core/rag/retriever.py](../app/core/rag/retriever.py) | 多路检索器（备选实现：Milvus Collection + BM25 + RRF） |
| [app/core/rag/reranker.py](../app/core/rag/reranker.py) | Cross-Encoder 重排序 |
| [app/core/rag/generator.py](../app/core/rag/generator.py) | 答案生成 + 引用标注解析 |
| [app/api/routes/chat.py](../app/api/routes/chat.py) | RAG API 路由（6 个端点） |

---

## 二、ETL 文档处理管线

### 2.1 文档解析（DocumentParser）

位于 [app/etl/parser.py](../app/etl/parser.py)。

**支持的格式**：

| 格式 | 解析方式 | MIME 类型 |
|------|----------|-----------|
| 纯文本 (.txt) | UTF-8 解码 | `text/*` |
| PDF | pypdf 逐页提取 | `application/pdf` |
| 其他 | 返回空文本 + warning 标记 | — |

**核心逻辑**：

```python
# parser.py:42-53
def parse_bytes(self, data: bytes, filename: str, mime_type: str | None) -> ParsedDocument:
    lower = filename.lower()

    # TXT：直接解码
    if lower.endswith(".txt") or (mime.startswith("text/") and mime != "text/html"):
        text = data.decode("utf-8", errors="ignore")
        return ParsedDocument(text=text[:self._max_chars], ...)  # ← 截断保护：最大 50 万字符

    # PDF：pypdf 逐页提取
    if lower.endswith(".pdf") or mime == "application/pdf":
        return self._parse_pdf(data, mime)

    # 不支持：静默返回空
    return ParsedDocument(text="", mime_type=mime, meta={"warning": "unsupported"})
```

**PDF 解析实现**：

```python
# parser.py:55-74
def _parse_pdf(self, data, mime):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")   # ← 单页失败不中断，继续下一页
    text = "\n".join(parts)[:self._max_chars]     # ← 全局截断
    return ParsedDocument(text=text, mime_type=mime, meta={"pages": str(len(reader.pages))})
```

**返回数据结构**：

```python
@dataclass
class ParsedDocument:
    text: str                         # 纯文本内容
    mime_type: str                    # 文件类型
    meta: dict[str, str]              # 元数据（如 {"pages": "5"}）
```

### 2.2 文档分块（DocumentChunker）

位于 [app/etl/chunker.py](../app/etl/chunker.py)。

**四种分块策略**：

| 策略 | 算法 | 适用场景 |
|------|------|----------|
| `fixed` | 固定窗口滑动，步长 = chunk_size - overlap | 简单快速 |
| `recursive`（默认） | `\n\n → \n → 。 → . → 空格` 优先级递归切分 | 通用中英文混合文档 |
| `paragraph` | 按空行分段 → 合并到目标长度 | 结构化文档（报告、论文） |
| `langchain` | LangChain RecursiveCharacterTextSplitter | 与 LangChain 生态对齐 |

**默认参数**：`chunk_size=512, chunk_overlap=64`

```python
# chunker.py:22-31
class DocumentChunker:
    def __init__(self, chunk_size=512, chunk_overlap=64, encoding_name="cl100k_base"):
        self.chunk_size = max(32, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        # tiktoken 精确 token 计数
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None   # ← 降级为字符计数
```

**参数选择理由**：

| 参数 | 值 | 理由 |
|------|-----|------|
| chunk_size | 512 | all-MiniLM-L6-v2 最大输入 256 token；512 字符块保证单块不超限且有足够语义 |
| chunk_overlap | 64（12.5%） | 防止关键信息恰好落在块边界上被切断 |
| encoding | cl100k_base | GPT-4 同款分词器，中文按字符近似但比纯字符计数准确 |

**递归分块核心逻辑**：

```python
# chunker.py:105-143
def _chunk_recursive(self, text, separators=None, *, _depth=0):
    if _depth > 24:
        return self._chunk_fixed(text)           # ← 递归过深 → 固定窗口兜底

    seps = separators or ["\n\n", "\n", "。", ". ", " "]

    if self._len(text) <= self.chunk_size:
        return [text]                             # ← 已足够短，直接返回

    for i, sep in enumerate(seps):
        if sep not in text:
            continue                              # ← 当前分隔符不存在，试下一个
        pieces = text.split(sep)
        merged = []
        buf = ""
        for part in pieces:
            candidate = (buf + sep + part) if buf else part
            if self._len(candidate) <= self.chunk_size:
                buf = candidate                   # ← 能合并就合并
            else:
                if buf:
                    merged.extend(self._chunk_recursive(buf, seps[i+1:], _depth=_depth+1))
                buf = part
        # ...
        return merged if merged else self._chunk_fixed(text)

    return self._chunk_fixed(text)                # ← 所有分隔符都不存在 → 固定窗口
```

**设计要点**：
- 分隔符有优先级：段落（`\n\n`）> 行（`\n`）> 句子（`。`）> 子句（`. `）> 词（空格）
- 当前分隔符切分后，剩余部分用下一级分隔符递归处理
- 递归深度上限 24，防止极端输入导致栈溢出
- 所有策略最终都有固定窗口兜底

### 2.3 ETL 流水线（ETLPipeline）

位于 [app/etl/pipeline.py](../app/etl/pipeline.py)。

组合解析和分块，并提供 `on_chunks` 回调钩子用于向量化入库：

```python
# pipeline.py:35-60
async def run_bytes(self, data, filename, mime_type, *, strategy=RECURSIVE, on_chunks=None):
    parsed = self._parser.parse_bytes(data, filename, mime_type)   # ① 解析
    chunks = self._chunker.chunk(parsed.text, strategy=strategy)   # ② 分块
    if on_chunks is not None:
        await on_chunks(chunks, parsed)                            # ③ 回调（向量化入库）
    return ETLResult(chunks=chunks, parsed=parsed, ...)
```

---

## 三、向量化与嵌入模型

### 3.1 主模型：sentence-transformers

位于 [app/core/rag_service.py:45-53](../app/core/rag_service.py#L45-L53)。

```python
def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # ← 懒加载，首次约 0.5s
    return _EMBEDDING_MODEL
```

**选型分析**：

| 维度 | all-MiniLM-L6-v2 | 备选（如 bge-large-zh） |
|------|------------------|------------------------|
| 向量维度 | 384 | 1024 |
| 模型大小 | ~80MB | ~1.3GB |
| 推理速度 | ~1400 sentences/sec (CPU) | ~200 sentences/sec (CPU) |
| 中文支持 | 多语言训练，基本可用 | 中文 SOTA |
| 部署成本 | 零（本地运行，无 API 费用） | 零（本地运行，但资源占用高） |

作为面试项目，384 维 + 轻量级是合理的默认选择。生产环境中文场景可能换 `BAAI/bge-large-zh-v1.5`。

### 3.2 降级模型：字符 n-gram 哈希

当 sentence-transformers 安装失败时，自动降级为纯 Python 实现：

```python
# rag_service.py:366-385
class _NGramEmbedding:
    """字符 n-gram 哈希嵌入（256 维，无需外部模型）。"""

    def embed_query(self, text):
        vec = [0.0] * 256
        text = text.lower().strip()
        for i in range(len(text)):
            for n in (2, 3):                    # ← 2-gram + 3-gram
                if i + n <= len(text):
                    h = hash(text[i:i+n]) % 256  # ← 哈希映射到 256 维
                    vec[h] += 1.0
        # L2 归一化
        norm = math.sqrt(sum(v*v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
```

**降级质量**：n-gram 嵌入比随机好（相同字符组合会映射到相近向量），但远不如语义嵌入。仅保证系统在无依赖环境下仍可运行。

### 3.3 批量和单条编码

```python
# rag_service.py:548-561
def _encode(self, text):
    if isinstance(self._embed, _NGramEmbedding):
        return self._embed.embed_query(text)          # ← 降级路径
    return self._embed.encode(text, normalize_embeddings=True).tolist()  # ← 主路径

def _encode_batch(self, texts):
    if isinstance(self._embed, _NGramEmbedding):
        return [self._embed.embed_query(t) for t in texts]
    vecs = self._embed.encode(texts, normalize_embeddings=True)  # ← 批量编码，底层 C 加速
    return [v.tolist() for v in vecs]
```

**性能要点**：`_encode_batch` 调用模型底层的批量编码（C 实现），比逐个编码快 5-10 倍。文档摄入时使用批量路径。

---

## 四、存储引擎（Milvus 主 + 内存降级）

### 4.1 架构设计

两种实现暴露完全相同的接口，调用方无需区分：

```python
# rag_service.py:474-505
class RAGService:
    def __init__(self):
        # 尝试 Milvus
        try:
            self._store = _MilvusRAGStore(db_path, dim=embed_dim)
            self._use_milvus = True
        except Exception:
            self._store = _InMemoryDocStore()     # ← 自动降级
            self._use_milvus = False
```

**统一接口**（两种实现都支持）：

| 方法 | 说明 |
|------|------|
| `add(doc_id, content, embedding, source)` | 插入单条 |
| `add_batch([(id, text, vec, src), ...])` | 批量插入 |
| `remove_by_source(source)` | 按来源删除 |
| `vector_search(query_vec, top_k)` | 向量检索 |
| `keyword_search(query_text, top_k)` | BM25 关键词检索 |
| `hybrid_search(query_vec, query_text, top_k)` | RRF 混合检索 |
| `list_sources()` | 列出所有文档来源 |

### 4.2 Milvus 主实现（_MilvusRAGStore）

位于 [app/core/rag_service.py:153-358](../app/core/rag_service.py#L153-L358)。

**初始化**：

```python
class _MilvusRAGStore:
    def __init__(self, db_path, dim=384):
        self._client = MilvusClient(db_path)       # ← Milvus Lite 本地文件
        self._collection = "rag_chunks"
        self._bm25 = _BM25()                       # ← 内存 BM25 索引（与向量库同步维护）
        self._chunk_to_source = {}

        if not self._client.has_collection(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                dimension=dim,
                metric_type="COSINE",              # ← 余弦相似度
                id_type="string",                  # ← 自定义 UUID 作为主键
                max_length=64,
            )
            self._client.load_collection(self._collection)
```

**Schema 设计**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | VARCHAR(64) | UUID 主键 |
| `vector` | FLOAT_VECTOR(384) | 语义向量 |
| `content` | VARCHAR | 文本内容 |
| `source` | VARCHAR | 来源文档名 |

**注意**：BM25 索引与 Milvus 是**并行维护**的——每次 `add` 同时写入 Milvus 和 BM25，每次 `remove_by_source` 同时删除两边。两者在物理上是独立的存储（Milvus 文件 + 内存 BM25），逻辑上保持一致。

**集合加载（_ensure_loaded）**：

```python
def _ensure_loaded(self):
    """Milvus Lite 在重启或空闲后会自动释放集合，需要重新加载。"""
    try:
        self._client.load_collection(self._collection)
    except Exception:
        pass  # 已加载状态下重复 load 不会报错
```

这是 Milvus Lite 的特有设计——数据持久化在文件，但索引在内存；空闲释放是省内存的默认行为。

### 4.3 内存降级实现（_InMemoryDocStore）

位于 [app/core/rag_service.py:388-460](../app/core/rag_service.py#L388-L460)。

```python
class _InMemoryDocStore:
    def __init__(self):
        self._docs: list[dict] = []          # ← Python list 存所有文档
        self._bm25 = _BM25()                 # ← 同样有 BM25 索引

    def vector_search(self, query_vec, top_k):
        # 暴力点积计算 + 排序
        scored = []
        for d in self._docs:
            dot = sum(a * b for a, b in zip(query_vec, d["embedding"]))
            scored.append((dot, d))
        scored.sort(key=lambda x: -x[0])
        return [RetrievalResult(...) for s, d in scored[:top_k]]
```

**与 Milvus 实现的功能等价性**：
- 接口完全一致
- 都支持 BM25 + 向量 + RRF 混合
- 区别仅在于向量检索使用 `O(n)` 暴力点积而非 `O(log n)` ANN 索引

### 4.4 BM25 实现

位于 [app/core/rag_service.py:66-145](../app/core/rag_service.py#L66-L145)。

**Okapi BM25 完整公式**：

```
BM25(d, q) = Σ IDF(q_i) · (f(q_i, d) · (k1 + 1)) / (f(q_i, d) + k1 · (1 - b + b · |d|/avgdl))

其中:
  IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))    # 平滑 IDF
  k1 = 1.5   # 词频饱和度参数
  b = 0.75   # 文档长度归一化参数
```

```python
class _BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self._ids = []               # doc_id 列表
        self._contents = []          # 原文列表
        self._doc_freqs = []         # 每文档的词频 dict
        self._doc_lens = []          # 每文档 token 数
        self._df = {}                # 全局文档频率
        self._N = 0                  # 文档总数
        self._idf = {}               # 预计算 IDF
        self._avgdl = 0.0            # 平均文档长度

    def search(self, query, top_k):
        q_terms = _tokenize(query)
        scores = []
        for i in range(self._N):
            s = 0.0
            for t in q_terms:
                idf = self._idf.get(t, 0.0)
                f = self._doc_freqs[i].get(t, 0)
                if f == 0:
                    continue
                dl = self._doc_lens[i]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if s > 0:
                scores.append((self._ids[i], s, self._contents[i]))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
```

**分词器**：简单正则分词，适合中英文混合场景：

```python
def _tokenize(text):
    """按字母-数字-中文连续段切分。"""
    return [t.lower() for t in re.findall(r"[\w一-鿿㐀-䶿]+", text) if t]
```

**IDF 平滑**：使用 `ln(1 + (N - df + 0.5) / (df + 0.5))` 而非标准 IDF——对低频词更平滑，避免 IDF 爆炸（当 df=1, N=10000 时标准 IDF=9.21，平滑 IDF=8.52）。

---

## 五、检索系统（三模式 + RRF 融合）

### 5.1 检索入口

```python
# rag_service.py:638-656
async def search(self, query, top_k=5, mode="hybrid"):
    if self._store.count == 0:
        return []

    mode = mode.lower()
    if mode == "vector":
        q_vec = self._encode(query)
        return self._store.vector_search(q_vec, top_k)
    elif mode == "keyword":
        return self._store.keyword_search(query, top_k)
    else:  # hybrid
        q_vec = self._encode(query)
        return self._store.hybrid_search(q_vec, query, top_k)
```

### 5.2 向量检索（vector）

```python
# rag_service.py:276-305（_MilvusRAGStore）
def vector_search(self, query_vec, top_k):
    self._ensure_loaded()
    raw = self._client.search(
        collection_name=self._collection,
        data=[query_vec],
        limit=top_k,
        output_fields=["content", "source"],
    )
    # 返回 [(id, distance_score, content), ...]
    # distance 是 COSINE 相似度（越大越好）
```

**关键细节**：Milvus COSINE 距离返回值范围 [-1, 1]，1 表示完全相同。这与 L2 距离（越小越好）方向相反，需要统一。

### 5.3 关键词检索（keyword）

委托给 BM25 索引：

```python
# rag_service.py:309-323
def keyword_search(self, query, top_k):
    ranked = self._bm25.search(query, top_k)
    return [RetrievalResult(id=doc_id, content=content, score=score, source="keyword")
            for doc_id, score, content in ranked]
```

### 5.4 RRF 混合融合（核心）

位于 [app/core/rag_service.py:327-353](../app/core/rag_service.py#L327-L353)。

**RRF 公式**（Reciprocal Rank Fusion）：

```
RRF_score(d) = Σ_{channel}  1 / (k + rank_in_channel + 1)
                               k = 60 (经验值)
```

**实现流程**：

```
Query
  │
  ├── 向量检索：取 top_k × 2 条 ──┐
  │                                │
  └── BM25 检索：取 top_k × 2 条 ──┤
                                   │
                          ┌────────▼────────┐
                          │    RRF 融合      │
                          │                  │
                          │  对每条文档 d：   │
                          │  score[d] =      │
                          │    1/(60+rank_vec+1)│
                          │  + 1/(60+rank_bm25+1)│
                          │                  │
                          │  按 score 降序   │
                          │  取 top_k 条     │
                          └────────┬────────┘
                                   │
                              最终结果
```

**源码**：

```python
def hybrid_search(self, query_vec, query_text, top_k):
    vec_results = self.vector_search(query_vec, top_k * 2)    # ← 各取双倍候选
    kw_results = self.keyword_search(query_text, top_k * 2)

    rrf_k = 60
    scores: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}

    # 向量路
    for rank, r in enumerate(vec_results):
        scores[r.id] = 1.0 / (rrf_k + rank + 1)
        best[r.id] = r

    # 关键词路（叠加到已有分数）
    for rank, r in enumerate(kw_results):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)
        if r.id not in best:
            best[r.id] = r       # ← 保留最高分的完整结果对象

    # 按融合分数降序取 top_k
    merged_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    merged = []
    for rid in merged_ids:
        item = best[rid]
        item.score = scores[rid]    # ← 覆盖为 RRF 分数
        item.source = "hybrid"
        merged.append(item)
    return merged
```

**k=60 的选择**：

| k 值 | 效果 |
|------|------|
| 0 | rank1 得分 1.0，rank2 得分 0.5……高排名优势过大 |
| 60 | rank1 得分 0.0164，rank2 得分 0.0161……排名间差异平滑 |
| ∞ | 所有排名得分趋同，退化为无序集合 |

60 来自 RRF 原始论文的实验结果——在多项 TREC 数据集上 k=60 取得最佳效果。k 越大，排名靠后的文档在融合中被赋予的相对权重越高。

**为什么取 top_k × 2 而非在全量上融合**：全量 RRF 意味着 BM25 要对所有文档排序（很快），但向量检索也要对全库做 ANN（慢）。取双倍候选在实践中几乎没有召回损失（某路排名靠后的文档不可能在融合中胜出），但计算量可控。

### 5.5 备选实现：MultiRetriever

位于 [app/core/rag/retriever.py](../app/core/rag/retriever.py)。

此模块提供了另一种接口风格的检索器，设计上可以对接外部 Milvus Collection（而非内置的 MilvusClient）：

```python
class MultiRetriever:
    """多路检索引擎：向量 + BM25 + 混合（RRF）。"""

    def __init__(self, milvus_client, embedding_model):
        self._milvus = milvus_client         # ← pymilvus Collection 对象
        self._embed = embedding_model         # ← embed_query 协议
        self._bm25 = _BM25Index()            # ← 独立的 BM25 索引
        self._id_to_text: dict[str, str] = {}
        self._rrf_k = 60

    async def retrieve(self, query, top_k=10, mode="hybrid"):
        if mode == "vector":   return await self.vector_search(query, top_k)
        if mode == "keyword":  return await self.keyword_search(query, top_k)
        return await self.hybrid_search(query, top_k)
```

**与 rag_service.py 中的 _MilvusRAGStore 的关键区别**：

| 维度 | _MilvusRAGStore + RAGService | MultiRetriever |
|------|------------------------------|----------------|
| Milvus 接口 | MilvusClient（Lite SDK） | Collection（ORM 风格） |
| BM25 管理 | 自动同步（add 时同时写入） | 手动同步（需调 register_keyword_documents） |
| 嵌入编码 | 内部管理（encode 方法） | 外部注入 |
| 使用场景 | 主要生产路径（RAGService 内部） | 备选路径（可独立用于其他场景） |

**注意**：当前项目的主路径是 `RAGService` → `_MilvusRAGStore`，`MultiRetriever` 作为独立的可复用组件存在，但实际 API 调用走的是 `RAGService`。

---

## 六、重排序（Cross-Encoder）

位于 [app/core/rag/reranker.py](../app/core/rag/reranker.py)。

### 6.1 为什么需要重排序

RRF 融合后的结果已经是两路综合排名，但 Bi-Encoder（embedding 模型）的语义匹配精度有限：

| 方法 | 精度 | 速度 | 说明 |
|------|------|------|------|
| Bi-Encoder（embedding） | 中 | 快 | Query 和 Doc 独立编码，点积相似度 |
| Cross-Encoder | 高 | 慢 | Query+Doc 拼接后联合编码，逐对打分 |

Cross-Encoder 需要为每个 (query, doc) 对做一次完整推理，不能预计算向量。因此它只对 RRF 返回的 top_k（通常 10-20 条）做精排，而非全库。

### 6.2 实现

```python
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self._model_name = model_name
        self._device = device
        self._model = None   # ← 懒加载

    async def rerank(self, query, documents, top_k=5):
        if not documents:
            return []

        def _sync_predict():
            model = self._load_model()
            pairs = [(query, d.content) for d in documents]   # ← (query, doc) 对
            raw = model.predict(pairs)                         # ← 逐对打分
            return list(raw.tolist() if hasattr(raw, "tolist") else raw)

        scores = await asyncio.to_thread(_sync_predict)        # ← 异步线程执行
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for doc, sc in ranked[:top_k]:
            new_doc = doc.model_copy(deep=True)
            new_doc.score = float(sc)
            new_doc.metadata = {**new_doc.metadata, "rerank_score": float(sc)}
            out.append(new_doc)
        return out
```

**模型选择**：`cross-encoder/ms-marco-MiniLM-L-6-v2`——微软 MS MARCO 排行榜上的经典轻量模型，384 维，与 embedding 模型同系列，推理速度快。

**注意**：当前 RAGService 的主路径中**没有集成 Reranker**。这是一个已实现但未接入的独立组件。面试中可以说"架构上预留了重排序能力，可在检索流水线中插入"。

---

## 七、答案生成与引用标注

位于 [app/core/rag/generator.py](../app/core/rag/generator.py)。

### 7.1 Prompt 构造

```python
# generator.py:43-77
def _build_messages(self, query, contexts, chat_history):
    # 1. 上下文编号
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        ctx_lines.append(f"[{i}] (id={c.id})\n{c.content}")

    # 2. 组装 user content
    user_content = (
        f"## 检索上下文\n{context_block}\n\n"
        f"## 历史对话\n{history_text}\n\n"
        f"## 用户问题\n{query}\n\n"
        "请作答并在必要时使用 [1]、[2] 引用上下文编号。"
    )

    return [
        {"role": "system", "content": "你是严谨的知识助手。仅根据提供的上下文作答；"
                                       "若上下文不足请说明。回答中引用来源时使用 [1]、[2] 等形式。"},
        {"role": "user", "content": user_content},
    ]
```

**关键约束**：
- **仅根据提供的上下文作答**：减少幻觉——没有检索到的信息不能编造
- **上下文不足请说明**：让用户知道知识边界，而不是猜测
- **[n] 引用格式**：便于后处理解析引用

### 7.2 答案生成

```python
# generator.py:107-135
async def generate(self, query, contexts, chat_history):
    messages = self._build_messages(query, contexts, chat_history)

    # 调用 LLM
    if isinstance(self._llm, RAGLLMProtocol):
        raw = await self._llm.ainvoke(messages)
    else:
        raw = await self._llm.ainvoke(messages)

    answer = self._parse_llm_output(raw).strip()
    citations = self._extract_citations(answer, contexts)
    return RAGResponse(answer=answer, citations=citations, raw_contexts=contexts, model=...)
```

**LLM 输出解析**（兼容多种返回格式）：

```python
def _parse_llm_output(self, raw):
    if raw is None:         return ""
    if isinstance(raw, str): return raw
    if hasattr(raw, "content"): return str(raw.content)  # ← LangChain AIMessage
    if isinstance(raw, dict) and "content" in raw: return str(raw["content"])
    return str(raw)
```

### 7.3 引用提取

```python
# generator.py:79-93
def _extract_citations(self, answer, contexts):
    refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]  # ← 正则提取所有 [n]
    citations = []
    seen = set()
    for idx in refs:
        if idx in seen or idx < 1 or idx > len(contexts):
            continue           # ← 去重 + 越界过滤
        seen.add(idx)
        r = contexts[idx - 1]
        snippet = r.content[:200] + ("..." if len(r.content) > 200 else "")
        citations.append(Citation(index=idx, result_id=r.id, snippet=snippet))
    return citations
```

**返回结果示例**：

```json
{
  "answer": "高血压患者应注意低盐饮食，每日盐摄入量不超过5克[1]。同时建议每周进行至少150分钟的中等强度有氧运动[2]。",
  "citations": [
    {"index": 1, "result_id": "abc123", "snippet": "高血压饮食指南建议每日盐摄入量控制在5克以内..."},
    {"index": 2, "result_id": "def456", "snippet": "美国心脏协会推荐每周150分钟中等强度运动..."}
  ],
  "raw_contexts": [...],
  "model": "gpt-4"
}
```

---

## 八、RAGService 顶层服务

位于 [app/core/rag_service.py:468-689](../app/core/rag_service.py#L468-L689)。

### 8.1 初始化流程

```
RAGService.__init__()
  │
  ├── ① 加载 Embedding 模型
  │     try: sentence-transformers (384-dim)
  │     except: _NGramEmbedding (256-dim)
  │
  ├── ② 初始化向量存储
  │     try: _MilvusRAGStore (Milvus Lite)
  │     except: _InMemoryDocStore (Python list)
  │
  ├── ③ 创建 LLM 适配器
  │     ModelRouter → _LLMAdapter(ainvoke)
  │
  └── ④ 初始化生成器
        RAGGenerator(llm, model_name)
```

**总降级链**：`sentence-transformers → n-gram ∣∣ Milvus → 内存`（两层独立降级，共 4 种组合，任意组合均可正常运行）。

### 8.2 文档摄入

**文本直入**：

```python
async def ingest_text(self, text, source="manual"):
    chunks = self._chunker.chunk(text)
    embeddings = self._encode_batch(chunks)           # ← 批量编码
    items = [(str(uuid.uuid4()), c, emb, source) for c, emb in zip(chunks, embeddings)]
    self._store.add_batch(items)                      # ← 批量写入
    return len(chunks)
```

**文件上传**：

```python
async def ingest_file(self, data, filename, mime_type):
    parsed = self._parser.parse_bytes(data, filename, mime_type)  # ① 解析
    chunks = self._chunker.chunk(parsed.text)                     # ② 分块
    embeddings = self._encode_batch(chunks)                       # ③ 批量编码
    items = [(str(uuid.uuid4()), c, emb, filename) for c, emb in zip(chunks, embeddings)]
    self._store.add_batch(items)                                  # ④ 批量写入
    return len(chunks)
```

**文档删除**：

```python
def delete_document(self, filename):
    n = self._store.remove_by_source(filename)    # ← 同时从 Milvus + BM25 删除
    del self._documents[filename]
    return True
```

### 8.3 RAG 对话

```python
async def chat(self, query, chat_history=None, mode="hybrid", top_k=5):
    if self._store.count == 0:
        return RAGResponse(answer="知识库为空，请先上传文档。", ...)

    contexts = await self.search(query, top_k=top_k, mode=mode)  # ① 检索
    response = await self._generator.generate(                   # ② 生成
        query=query, contexts=contexts, chat_history=chat_history,
    )
    return response
```

---

## 九、API 路由层

位于 [app/api/routes/chat.py:191-326](../app/api/routes/chat.py#L191-L326)。

### 9.1 端点一览

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/chat/rag` | RAG 增强对话 |
| `POST` | `/rag/ingest` | 文档上传入库 |
| `GET` | `/rag/status` | 知识库状态（块数） |
| `GET` | `/rag/documents` | 文档列表 |
| `DELETE` | `/rag/documents/{filename}` | 删除文档 |
| `POST` | `/rag/search_compare` | 三种检索模式并行对比（调试用） |

### 9.2 RAG 与 Agent 的集成

RAGService 被注入到 AgentService 中，供 ModeRouter 判断知识库状态：

```python
# chat.py:43-52
def _get_agent():
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
        try:
            _agent_service.set_rag_service(_get_rag())  # ← 注入 RAG 引用
        except Exception:
            pass
    return _agent_service
```

ModeRouter 利用知识库状态影响路由决策：

```python
# agent_service.py:333-339
kb_available = False
if self._rag_service is not None:
    kb_available = self._rag_service.chunks_count > 0
self._mode_router.kb_available = kb_available
# → LLM Router prompt 中包含"知识库可用: 是/否"
```

### 9.3 搜索对比端点（调试利器）

```python
# chat.py:289-325
@router.post("/rag/search_compare")
async def rag_search_compare(req):
    # 并行执行三种模式
    vec_task = rag.search(query, top_k=top_k, mode="vector")
    kw_task = rag.search(query, top_k=top_k, mode="keyword")
    hyb_task = rag.search(query, top_k=top_k, mode="hybrid")
    vec, kw, hyb = await asyncio.gather(vec_task, kw_task, hyb_task)

    # 返回三种模式的对比结果
    return {"vector": _fmt(vec), "keyword": _fmt(kw), "hybrid": _fmt(hyb), ...}
```

这个端点可以直观对比三种检索模式的差异——同一查询下，BM25 命中了哪些词，向量找到了哪些语义相近但关键词不匹配的内容，RRF 如何融合两者。

---

## 十、完整请求链路

### 链路 A：文档上传

```
POST /rag/ingest (PDF 文件)
        │
        ▼
RAGService.ingest_file(data, filename, mime_type)
        │
        ├── ① DocumentParser.parse_bytes()
        │      └── pypdf 逐页提取 → 纯文本
        │
        ├── ② DocumentChunker.chunk(text)
        │      └── 递归分块（\n\n → \n → 。）→ N 个文本块
        │
        ├── ③ _encode_batch(chunks)
        │      └── SentenceTransformer.encode(批量) → 归一化向量
        │
        └── ④ _MilvusRAGStore.add_batch(items)
               ├── MilvusClient.insert(collection, data)  → 向量入库
               ├── _BM25.add(doc_id, content) 每个块     → 关键词索引更新
               └── _chunk_to_source[doc_id] = filename   → 来源映射
```

### 链路 B：RAG 对话

```
POST /chat/rag {"messages": [...], "query": "高血压饮食建议"}
        │
        ▼
RAGService.chat(query="高血压饮食建议", mode="hybrid", top_k=5)
        │
        ├── ① search(query, top_k=5, mode="hybrid")
        │      │
        │      ├── _encode("高血压饮食建议") → 384 维向量
        │      │
        │      ├── vector_search(query_vec, top_k=10)
        │      │      └── MilvusClient.search(COSINE) → 10 条语义相近的块
        │      │
        │      ├── keyword_search("高血压饮食建议", top_k=10)
        │      │      └── _BM25.search() → 10 条关键词匹配的块
        │      │
        │      └── RRF 融合 → 综合排名取 top_k=5
        │             score = 1/(60+rank_vec+1) + 1/(60+rank_bm25+1)
        │
        └── ② RAGGenerator.generate(query, contexts, chat_history)
               │
               ├── _build_messages()
               │      └── System: "仅根据上下文作答，引用用 [n]"
               │      └── User: "## 检索上下文\n[1] xxx\n[2] yyy\n\n## 用户问题\n..."
               │
               ├── LLM.ainvoke(messages) → "高血压患者应注意...[1]..."
               │
               └── _extract_citations(answer, contexts)
                      └── 正则提取 [1] [2] → Citation 对象列表
```

### 链路 C：搜索对比（调试）

```
POST /rag/search_compare {"query": "糖尿病", "top_k": 5}
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  asyncio.gather(三种模式并行)                           │
│                                                        │
│  vector_search("糖尿病", 5)    → [semantic results]    │
│  keyword_search("糖尿病", 5)   → [BM25 results]        │
│  hybrid_search("糖尿病", 5)    → [RRF results]         │
│                                                        │
│  总耗时: ~100ms（取决于最慢的一路）                      │
└────────────────────────────────────────────────────────┘
        │
        ▼
返回对比 JSON:
{
  "elapsed_seconds": 0.098,
  "vector":  [{"rank":1, "content":"...", "score":0.92}, ...],
  "keyword": [{"rank":1, "content":"...", "score":8.54}, ...],
  "hybrid":  [{"rank":1, "content":"...", "score":0.031}, ...]
}
```

---

## 十一、关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 嵌入模型 | all-MiniLM-L6-v2 (384-dim) | 轻量本地运行，CPU 友好，多语言支持，零成本 |
| 向量数据库 | Milvus Lite | 本地文件持久化，COSINE 相似度，零配置部署 |
| 关键词检索 | 自研 Okapi BM25 | 对精确术语匹配不可或缺，与向量检索互补 |
| 融合算法 | RRF (k=60) | 无需调参，跨通道分数天然不可比时适用，论文验证成熟 |
| 分块策略 | 递归分块（默认 512/64） | 适配中文句段结构，递归深度有限，token 精确计数 |
| 嵌入降级 | 字符 n-gram 哈希 (256-dim) | 零依赖保底，确保无外部模型时系统可用 |
| 存储降级 | Python list + 暴力搜索 | 零依赖保底，接口完全兼容 Milvus |
| 引用机制 | [n] 标记 + 后正则提取 | 简单可靠，LLM 天然理解编号格式，提取零 token 消耗 |
| 重排序 | Cross-Encoder（已实现未集成） | 预留架构位置，可在检索后插入精排 |
| System Prompt | "仅根据上下文作答，不足请说明" | 减少幻觉，明确知识边界 |

### 降级全景

```
Embedding 模型
  ├── sentence-transformers (384-dim)  ← 主路径
  └── _NGramEmbedding (256-dim)        ← 降级

存储引擎
  ├── _MilvusRAGStore (Milvus Lite)    ← 主路径
  └── _InMemoryDocStore (Python list)  ← 降级

检索模式
  ├── hybrid (vector + BM25 + RRF)     ← 默认
  ├── vector                           ← 纯语义
  └── keyword                          ← 纯关键词

四种降级组合任意可用：
  sentence-transformers + Milvus    ✅ 全功能
  sentence-transformers + 内存       ✅ 语义检索可用
  n-gram + Milvus                   ✅ 关键词 + 弱语义
  n-gram + 内存                     ✅ 最基础功能（零依赖可运行）
```
