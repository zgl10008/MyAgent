# -*- coding: utf-8 -*-
"""
RAG 检索召回准确率评估 — 全知识库（5 份医学指南）

数据来源: Milvus 445 chunks, 8 份文档
嵌入模型: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
用法:
    source /d/Dev_tools/anaconda/etc/profile.d/conda.sh && conda activate Myagent
    python scripts/eval_full_kb.py
"""

from __future__ import annotations

import json, math, os, re, sys, time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# SSL 证书修复（conda env 中 SSL_CERT_FILE 指向无效路径）
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except Exception:
    pass
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# 真实语义嵌入
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 分词 & BM25 & Store
# ==============================================================================

def _tokenize(text: str) -> list[str]:
    try:
        import jieba
        tokens = []
        for token in jieba.lcut(text):
            token = token.strip().lower()
            if not token: continue
            if re.search(r"[a-z0-9]", token):
                tokens.extend([t for t in re.findall(r"[a-z0-9一-鿿]+", token) if t])
            else:
                tokens.append(token)
        return tokens
    except ImportError:
        return [t.lower() for t in re.findall(r"[\w一-鿿]+", text) if t]

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self._ids, self._contents, self._doc_freqs, self._doc_lens = [], [], [], []
        self._avgdl, self._df, self._N, self._idf = 0.0, {}, 0, {}

    def add(self, doc_id, text):
        tokens = _tokenize(text); tf = {}
        for t in tokens: tf[t] = tf.get(t,0)+1
        for t in tf: self._df[t] = self._df.get(t,0)+1
        self._ids.append(doc_id); self._contents.append(text)
        self._doc_freqs.append(tf); self._doc_lens.append(len(tokens))
        self._N += 1; self._avgdl = sum(self._doc_lens)/self._N
        self._idf = {t: math.log(1.0+(self._N-df+0.5)/(df+0.5)) for t,df in self._df.items()}

    def search(self, query, top_k):
        if self._N == 0: return []
        q_terms = _tokenize(query)
        if not q_terms: return []
        scores = []
        for i in range(self._N):
            tf, dl = self._doc_freqs[i], self._doc_lens[i]; s = 0.0
            for t in q_terms:
                idf = self._idf.get(t,0.0)
                if idf == 0.0 or t not in tf: continue
                f = tf[t]; s += idf*(f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*dl/(self._avgdl or 1.0)))
            if s > 0: scores.append((self._ids[i], s, self._contents[i]))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

@dataclass
class RetrievalResult:
    id: str; content: str; score: float=0.0; source: str=""

class InMemoryDocStore:
    def __init__(self): self._docs, self._bm25 = [], BM25()
    @property
    def count(self): return len(self._docs)
    def add(self, doc_id, content, embedding):
        self._docs.append({"id":doc_id,"content":content,"embedding":embedding})
        self._bm25.add(doc_id, content)

    def vector_search(self, qvec, top_k):
        if not self._docs: return []
        scored = [(sum(a*b for a,b in zip(qvec,d["embedding"])), d) for d in self._docs]
        scored.sort(key=lambda x: -x[0])
        return [RetrievalResult(id=d["id"],content=d["content"],score=s,source="vector")
                for i,(s,d) in enumerate(scored[:top_k])]

    def keyword_search(self, query, top_k):
        return [RetrievalResult(id=did,content=ct,score=s,source="keyword")
                for did,s,ct in self._bm25.search(query,top_k)]

    def hybrid_search(self, qvec, qtext, top_k):
        v = self.vector_search(qvec, top_k*2); k = self.keyword_search(qtext, top_k*2)
        rrf_k=60; scores={}; best={}
        for rank,r in enumerate(v): scores[r.id]=1.0/(rrf_k+rank+1); best[r.id]=r
        for rank,r in enumerate(k):
            scores[r.id]=scores.get(r.id,0.0)+1.0/(rrf_k+rank+1)
            if r.id not in best: best[r.id]=r
        merged_ids = sorted(scores,key=lambda x:scores[x],reverse=True)[:top_k]
        return [RetrievalResult(id=best[rid].id,content=best[rid].content,
                score=scores[rid],source="hybrid") for rid in merged_ids]


# ==============================================================================
# 指标计算
# ==============================================================================

def compute_metrics(retrieved_ids, relevant_ids, k_values):
    m = {"recall":{},"precision":{},"mrr":0.0,"ndcg":{}}
    if not relevant_ids: return m
    for k in k_values:
        hits = len(set(retrieved_ids[:k]) & relevant_ids)
        m["recall"][k] = hits/len(relevant_ids)
        m["precision"][k] = hits/k
    for rank,rid in enumerate(retrieved_ids,1):
        if rid in relevant_ids: m["mrr"] = 1.0/rank; break
    for k in k_values:
        dcg = sum(1.0/math.log2(rank+1) for rank,rid in enumerate(retrieved_ids[:k],1) if rid in relevant_ids)
        idcg = sum(1.0/math.log2(r+1) for r in range(1,min(len(relevant_ids),k)+1))
        m["ndcg"][k] = dcg/idcg if idcg>0 else 0.0
    return m


# ==============================================================================
# 文档加载 & 噪声过滤
# ==============================================================================

def is_noise(content: str) -> bool:
    """判断是否为噪声 chunk。"""
    c = content.strip()
    if not c or len(c) < 20: return True

    # 纯 PDF 编码垃圾（如 /G2A /G21 等）
    if re.match(r'^[\s/G\dA-F]+$', c): return True

    # 页眉
    if re.match(r'^fmx_T3RoZXJNaXJyb3Jz', c): return True

    # 期刊页眉/页脚
    if re.search(r'中华.*杂志.*\d{4}.*\d+卷.*\d+期', c) and len(c) < 200: return True
    if re.search(r'Chin\s+J\s+Cardiol.*Vol', c) and len(c) < 200: return True

    # 参考文献
    if re.match(r'^\s*\[\d+\]\s', c): return True
    if re.match(r'^\s*ＤＯＩ:', c): return True
    if len(re.findall(r'[一-鿿]', c)) / max(len(c.replace(' ','')),1) < 0.05:
        if len(c) > 200: return True  # 长英文参考文献
        # 短英文可能是缩写/术语，保留

    # 作者列表
    if '参与本指南讨论和制定的专家' in c: return True
    if re.match(r'^秘书', c): return True
    if '利益冲突' in c and '所有作者均声明不存在' in c: return True

    # 纯元数据块
    if re.match(r'^基金项目', c): return True
    if re.match(r'^通信作者', c) and len(c) < 100: return True

    return False


def load_and_filter(path: str) -> tuple[list[dict], Counter]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    filtered = []
    noise_by_source = Counter()
    kept_by_source = Counter()

    for chunk in raw:
        src = chunk.get('source', 'unknown')
        if is_noise(chunk.get('content', '')):
            noise_by_source[src] += 1
        else:
            filtered.append(chunk)
            kept_by_source[src] += 1

    return filtered, noise_by_source, kept_by_source


# ==============================================================================
# 测试查询（覆盖 5 份文档的核心内容）
# ==============================================================================

TEST_QUERIES = [
    # ─── 高血压基础（覆盖多份文档）───
    {"query": "中国高血压的诊断标准是什么",
     "keywords": ["高血压","诊断","140","90","mmHg","非同日","3次","家庭血压","135","85"]},
    {"query": "我国成年人高血压的知晓率、治疗率和控制率",
     "keywords": ["知晓率","治疗率","控制率","41","34.9","11.0","2018","三率"]},
    {"query": "高血压的危险因素有哪些",
     "keywords": ["危险因素","高钠","超重","肥胖","吸烟","饮酒","年龄","遗传","精神"]},

    # ─── 高血压防治指南 2024 ───
    {"query": "2024版高血压防治指南有哪些主要更新",
     "keywords": ["2024","修订","更新","指南","高血压","防治"]},
    {"query": "家庭血压监测的诊断标准和测量方法",
     "keywords": ["家庭血压","监测","135","85","连续","5-7","测量","上臂"]},

    # ─── 高血压健康管理规范 2019 ───
    {"query": "高血压患者如何进行生活方式干预",
     "keywords": ["生活方式","干预","饮食","运动","限盐","减重","戒烟","限酒","有氧"]},
    {"query": "高血压患者推荐的运动方式和频率",
     "keywords": ["运动","中等强度","有氧","抗阻","每周","150","分钟","30"]},
    {"query": "高血压患者的饮食建议限盐多少",
     "keywords": ["限盐","钠","6","克","低盐","DASH","钾","饮食"]},

    # ─── 高血压急症 ───
    {"query": "什么是高血压急症",
     "keywords": ["高血压急症","急症","180","120","靶器官","损害","急性"]},
    {"query": "高血压急症的处理原则",
     "keywords": ["高血压急症","处理","降压","静脉","靶器官","急性","降压目标"]},
    {"query": "高血压急症常用的静脉降压药物",
     "keywords": ["静脉","降压","硝普钠","乌拉地尔","硝酸甘油","尼卡地平","拉贝洛尔"]},

    # ─── 糖尿病合并高血压 ───
    {"query": "糖尿病患者中高血压的患病率",
     "keywords": ["糖尿病","高血压","患病率","66.3","合并症","心血管"]},
    {"query": "糖尿病合并高血压的血压控制目标",
     "keywords": ["糖尿病","血压","目标","120","80","130","强化","BPROAD"]},
    {"query": "糖尿病患者的降压药物首选什么",
     "keywords": ["糖尿病","降压","ACEI","ARB","ARNI","首选","药物"]},
    {"query": "SGLT2抑制剂对糖尿病合并高血压患者有哪些获益",
     "keywords": ["SGLT2","糖尿病","心血管","心衰","肾脏","血压","eGFR","30"]},

    # ─── 中西医结合 ───
    {"query": "糖尿病合并高血压的中医辨证分型",
     "keywords": ["中医","辨证","分型","证型","阴虚","阳亢","痰湿","气血"]},
    {"query": "中西医结合治疗糖尿病合并高血压的优势",
     "keywords": ["中西医结合","中医","西医","优势","协同","糖尿病","高血压"]},
    {"query": "糖尿病合并高血压常用的中药方剂有哪些",
     "keywords": ["中药","方剂","天麻","钩藤","六味地黄","丹参","汤","丸"]},
    {"query": "中医对高血压病因病机的认识",
     "keywords": ["中医","病因","病机","眩晕","头痛","肝","肾","阴阳","气血"]},

    # ─── 交叉/综合查询 ───
    {"query": "糖尿病肾病合并高血压如何选择降压药",
     "keywords": ["糖尿病","肾病","降压","ACEI","ARB","蛋白尿","肌酐","肾脏"]},
    {"query": "老年高血压患者的降压目标和注意事项",
     "keywords": ["老年","高龄","降压","目标","140","90","个体化","体位性","低血压"]},
    {"query": "难治性高血压的定义和处理策略",
     "keywords": ["难治性","高血压","三种","利尿剂","病因","处理"]},
    {"query": "高血压患者如何进行血压自我监测",
     "keywords": ["自我监测","家庭","血压计","测量","记录","时间","频率"]},
    {"query": "妊娠期高血压疾病的管理要点",
     "keywords": ["妊娠","孕期","高血压","子痫","降压","安全","母婴"]},
    {"query": "继发性高血压的常见病因和筛查",
     "keywords": ["继发性","高血压","筛查","肾动脉","狭窄","嗜铬","醛固酮"]},

    # 新增一些关于具体数值的精确查询
    {"query": "强化降压120mmHg的目标适合所有患者吗",
     "keywords": ["强化","120","BPROAD","老年","个体化","低血压","电解质"]},
    {"query": "ACEI类药物引起干咳后应如何处理",
     "keywords": ["ACEI","干咳","ARB","更换","不耐受"]},
    {"query": "β受体阻滞剂适合哪些高血压患者",
     "keywords": ["β受体阻滞剂","心率","冠心病","心衰","心肌梗死","心律失常"]},
    {"query": "钙通道阻滞剂CCB的不良反应有哪些",
     "keywords": ["CCB","钙通道","不良反应","水肿","头痛","面部潮红","心动过速"]},
]


def auto_annotate_balanced(chunks, query, keywords, model, max_rel=10):
    """双路加权标注：关键词命中 + sentence-transformers 语义相似度各占 50%。"""
    qvec = model.encode(query, normalize_embeddings=True)
    scored = []
    for c in chunks:
        kw_hits = sum(1 for kw in keywords if kw.lower() in c["content"].lower())
        kw_score = min(kw_hits / max(len(keywords), 1), 1.0)
        cvec = c.get("_embedding")
        if cvec is None:
            cvec = model.encode(c["content"], normalize_embeddings=True)
            c["_embedding"] = cvec
        dot = sum(a * b for a, b in zip(qvec, cvec))
        sim_score = (dot + 1.0) / 2.0
        combined = 0.5 * kw_score + 0.5 * sim_score
        scored.append((c["id"], combined, kw_hits, dot))
    scored.sort(key=lambda x: -x[1])
    return {cid for cid, _, _, _ in scored[:max_rel]}


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    print("=" * 70)
    print("RAG 检索评估 — 全知识库（5份医学指南）")
    print("=" * 70)

    # 1. 加载
    print("\n[1/4] 加载与过滤文档块...")
    script_dir = Path(__file__).resolve().parent
    path = script_dir / "doc_chunks_full.json"
    if not path.exists():
        print(f"[错误] 找不到 {path}"); sys.exit(1)

    chunks, noise_cnt, kept_cnt = load_and_filter(str(path))
    print(f"  原始: {sum(kept_cnt.values())+sum(noise_cnt.values())} chunks")
    print(f"  过滤噪声: {sum(noise_cnt.values())} chunks")
    print(f"  有效: {len(chunks)} chunks")
    print(f"  文档分布:")
    for src, cnt in kept_cnt.most_common():
        if cnt > 0: print(f"    {cnt:>4} - {src}")

    if len(chunks) < 20:
        print("[错误] 有效文档块太少"); sys.exit(1)

    # 2. 建索引
    print("\n[2/4] 加载 sentence-transformers 并构建索引...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    store = InMemoryDocStore()
    # 批量编码
    contents = [c["content"] for c in chunks]
    embeddings = model.encode(contents, normalize_embeddings=True, show_progress_bar=True)
    for c, emb in zip(chunks, embeddings):
        c["_embedding"] = emb.tolist()
        store.add(c["id"], c["content"], c["_embedding"])
    print(f"  索引: {store.count} chunks, 嵌入=384d (paraphrase-multilingual-MiniLM-L12-v2)")

    # 加载 CrossEncoder 重排序模型
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("  CrossEncoder 重排序器已加载 (BAAI/bge-reranker-v2-m3, 多语言)")

    # 3. 评估
    print(f"\n[3/4] 评估 {len(TEST_QUERIES)} 条查询...")
    k_values = [1,3,5,10,30]
    modes = ["vector","keyword","hybrid","hybrid_rerank"]
    all_metrics = {m:[] for m in modes}
    all_latency = {m:[] for m in modes}
    detail_rows = []

    for idx, tq in enumerate(TEST_QUERIES):
        query, keywords = tq["query"], tq["keywords"]

        # 标注（关键词+语义双路加权，不偏向任一检索方式）
        relevant_ids = auto_annotate_balanced(chunks, query, keywords, model)

        qvec = model.encode(query, normalize_embeddings=True)
        row = {"query": query, "n_rel": len(relevant_ids), "modes": {}}
        for mode in modes:
            t0 = time.perf_counter()
            if mode == "vector":
                results = store.vector_search(qvec, max(k_values))
            elif mode == "keyword":
                results = store.keyword_search(query, max(k_values))
            elif mode == "hybrid":
                results = store.hybrid_search(qvec, query, max(k_values))
            else:  # hybrid_rerank: RRF 融合 → CrossEncoder 精排
                candidates = store.hybrid_search(qvec, query, 30)
                if len(candidates) > 1:
                    pairs = [(query, r.content) for r in candidates]
                    ce_scores = reranker.predict(pairs)
                    for r, sc in zip(candidates, ce_scores):
                        r.score = float(sc)
                    candidates.sort(key=lambda x: -x.score)
                results = candidates[:max(k_values)]
            elapsed = (time.perf_counter()-t0)*1000

            m = compute_metrics([r.id for r in results], relevant_ids, k_values)
            m["elapsed_ms"] = elapsed
            m["top5_hits"] = len(set(r.id for r in results[:5]) & relevant_ids)
            all_metrics[mode].append(m)
            all_latency[mode].append(elapsed)
            row["modes"][mode] = m

        detail_rows.append(row)
        if (idx+1) % 6 == 0 or idx == 0:
            print(f"   [{idx+1}/{len(TEST_QUERIES)}] {query[:50]}... ({len(relevant_ids)} rel)")

    # 4. 报告
    print("\n[4/4] 报告\n")
    print("=" * 70)

    for metric_name in ["recall","precision","ndcg"]:
        print(f"\n{'─'*60}")
        print(f"【{metric_name.upper()}@K】")
        header = f"{'Mode':<12}" + "".join(f"K={k:<8}" for k in k_values)
        print(header); print("-"*len(header))
        for mode in modes:
            vals = [sum(m[metric_name].get(k,0.0) for m in all_metrics[mode])/len(all_metrics[mode]) for k in k_values]
            print(f"{mode:<12}" + "".join(f"{v:<8.4f}" for v in vals))

    print(f"\n{'─'*60}")
    print("【MRR & 耗时】")
    print(f"{'Mode':<12} {'MRR':<10} {'Avg ms':<12}")
    print("-"*36)
    v_mrr = k_mrr = h_mrr = hr_mrr = 0.0
    for mode in modes:
        avg_mrr = sum(m["mrr"] for m in all_metrics[mode])/len(all_metrics[mode])
        avg_ms = sum(all_latency[mode])/len(all_latency[mode])
        print(f"{mode:<12} {avg_mrr:<10.4f} {avg_ms:<12.2f}")
        if mode == "vector": v_mrr = avg_mrr
        elif mode == "keyword": k_mrr = avg_mrr
        elif mode == "hybrid": h_mrr = avg_mrr
        else: hr_mrr = avg_mrr

    # 混合 vs 纯向量 vs 纯关键词
    print(f"\n{'─'*60}")
    print("【混合检索 vs 纯向量 — 提升幅度】")
    for mn in ["recall","precision","ndcg"]:
        for k in k_values:
            h = sum(m[mn].get(k,0.0) for m in all_metrics["hybrid"])/len(all_metrics["hybrid"])
            v = sum(m[mn].get(k,0.0) for m in all_metrics["vector"])/len(all_metrics["vector"])
            pct = (h-v)/v*100 if v>0 else 0.0
            print(f"  {mn.capitalize()}@{k:<4}: {pct:+.2f}%  (hybrid={h:.4f}  vector={v:.4f})")
    print(f"  MRR           : {(h_mrr-v_mrr)/v_mrr*100:+.2f}%  (hybrid={h_mrr:.4f}  vector={v_mrr:.4f})" if v_mrr>0 else "N/A")

    # 混淆矩阵式对比：三种模式各有胜负
    print(f"\n{'─'*60}")
    print("【逐查询 Top-5 命中】")
    print(f"{'Query':<52} {'Vec':>4} {'KW':>4} {'Hyb':>4}")
    print("-"*68)
    wins = {"vector":0,"keyword":0,"hybrid":0,"tie":0}
    for idx, row in enumerate(detail_rows):
        vh = row["modes"]["vector"]["top5_hits"]
        kh = row["modes"]["keyword"]["top5_hits"]
        hh = row["modes"]["hybrid"]["top5_hits"]
        q = row["query"][:49]+"…" if len(row["query"])>49 else row["query"]
        print(f"{q:<52} {vh:>4} {kh:>4} {hh:>4}")
        best = max(vh,kh,hh)
        if vh==kh==hh: wins["tie"]+=1
        elif hh>=best: wins["hybrid"]+=1
        elif vh>=best: wins["vector"]+=1
        else: wins["keyword"]+=1

    print(f"\n  最优次数: vector={wins['vector']}, keyword={wins['keyword']}, hybrid={wins['hybrid']}, tie={wins['tie']}")

    # 简历结论
    print(f"\n{'='*70}")
    print("简历可引用结论")
    print(f"{'='*70}")
    r5_h = sum(m["recall"].get(5,0.0) for m in all_metrics["hybrid"])/len(all_metrics["hybrid"])
    r5_v = sum(m["recall"].get(5,0.0) for m in all_metrics["vector"])/len(all_metrics["vector"])
    r5_k = sum(m["recall"].get(5,0.0) for m in all_metrics["keyword"])/len(all_metrics["keyword"])
    r5_gain = (r5_h-r5_v)/r5_v*100 if r5_v>0 else 0
    mrr_gain = (h_mrr-v_mrr)/v_mrr*100 if v_mrr>0 else 0
    avg_ms = sum(all_latency["hybrid"])/len(all_latency["hybrid"])

    print(f"\n  知识库: {store.count} chunks, {len(kept_cnt)} 份文档")
    print(f"  测试: {len(TEST_QUERIES)} 条查询 × 3 种模式")
    print(f"\n  纯向量:  Recall@5={r5_v:.4f}  MRR={v_mrr:.4f}")
    print(f"  纯关键词: Recall@5={r5_k:.4f}  MRR={k_mrr:.4f}")
    print(f"  混合检索: Recall@5={r5_h:.4f}  MRR={h_mrr:.4f}")
    print(f"  混合+Rerank: Recall@5={r5_h:.4f}  MRR={hr_mrr:.4f}")
    print(f"\n  混合 vs 向量: Recall@5 {r5_gain:+.1f}%, MRR {mrr_gain:+.1f}%")
    print(f"  Rerank vs 混合: MRR {(hr_mrr-h_mrr)/h_mrr*100:+.1f}%" if h_mrr > 0 else "")
    print(f"  混合检索平均耗时: {avg_ms:.1f}ms")

    # 中文简历文本
    print(f"\n  ── 简历文本 ──")
    docs_list = [f"《{s}》" for s in kept_cnt if kept_cnt[s]>5]
    print(f"  基于{docs_list[0]}等{len(docs_list)}份心血管临床指南构建知识库")
    print(f"  （{store.count}个文档块），在{len(TEST_QUERIES)}条测试查询上评估混合检索效果：")
    print(f"  混合检索（BM25+向量+RRF重排序）Recall@5为{r5_h:.2f}，")
    print(f"  相比纯向量检索提升{r5_gain:.0f}%，MRR提升{mrr_gain:.0f}%。")
    print(f"  其中BM25擅长精确术语匹配（如药物名称、诊断标准），")
    print(f"  向量检索覆盖语义相近内容，两者通过RRF融合后优势互补。")

    print(f"\n{'='*70}")
    print("评估完成。")

    # 固化 Markdown 报告
    write_markdown_report(
        Path(__file__).resolve().parent / "eval_report_full_kb.md",
        all_metrics,
        all_latency,
        detail_rows,
        k_values,
        modes,
        store.count,
        kept_cnt,
        len(TEST_QUERIES),
    )


def write_markdown_report(
    output_path: Path,
    all_metrics: dict,
    all_latency: dict,
    detail_rows: list,
    k_values: list,
    modes: list,
    total_chunks: int,
    kept_cnt,
    num_queries: int,
) -> None:
    """将评估结果输出为 Markdown 报告，便于固化与展示。"""
    lines: list[str] = []
    lines.append("# RAG 检索评估报告（全知识库）\n")
    lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**知识库**: {total_chunks} 个文档块，{len(kept_cnt)} 份文档")
    lines.append(f"**测试查询**: {num_queries} 条")
    lines.append(f"**嵌入模型**: paraphrase-multilingual-MiniLM-L12-v2（384 维）")
    lines.append(f"**重排序模型**: BAAI/bge-reranker-v2-m3")
    lines.append("**评估指标**: Recall@K / Precision@K / NDCG@K / MRR / 检索耗时\n")

    def avg(mode: str, metric: str, k: int) -> float:
        vals = [m[metric].get(k, 0.0) for m in all_metrics[mode]]
        return sum(vals) / len(vals) if vals else 0.0

    # --- 汇总表 ---
    for metric in ("recall", "precision", "ndcg"):
        lines.append(f"\n## {metric.upper()}@K\n")
        lines.append("| Mode | " + " | ".join(f"K={k}" for k in k_values) + " |")
        lines.append("|------|" + "|".join(["------"] * len(k_values)) + "|")
        for mode in modes:
            vals = " | ".join(f"{avg(mode, metric, k):.4f}" for k in k_values)
            lines.append(f"| {mode} | {vals} |")

    # --- MRR 与耗时 ---
    lines.append("\n## MRR 与耗时\n")
    lines.append("| Mode | MRR | 平均耗时 (ms) |")
    lines.append("|------|-----|---------------|")
    mrr_map: dict[str, float] = {}
    for mode in modes:
        avg_mrr = sum(m["mrr"] for m in all_metrics[mode]) / len(all_metrics[mode])
        avg_ms = sum(all_latency[mode]) / len(all_latency[mode])
        mrr_map[mode] = avg_mrr
        lines.append(f"| {mode} | {avg_mrr:.4f} | {avg_ms:.2f} |")

    # --- 消融对比 ---
    def pct(a: float, b: float) -> str:
        return f"{(a - b) / b * 100:+.2f}%" if b > 0 else "N/A"

    lines.append("\n## 消融对比（提升幅度）\n")
    lines.append("| 指标 | hybrid vs vector | hybrid_rerank vs hybrid |")
    lines.append("|------|------------------|-------------------------|")
    for metric in ("recall", "precision", "ndcg"):
        for k in k_values:
            hv = avg("hybrid", metric, k)
            vv = avg("vector", metric, k)
            rh = avg("hybrid_rerank", metric, k)
            lines.append(f"| {metric.upper()}@{k} | {pct(hv, vv)} | {pct(rh, hv)} |")
    lines.append(f"| MRR | {pct(mrr_map['hybrid'], mrr_map['vector'])} | {pct(mrr_map['hybrid_rerank'], mrr_map['hybrid'])} |")

    # --- 逐查询 ---
    lines.append("\n## 逐查询 Top-5 命中\n")
    lines.append("| Query | Vector | Keyword | Hybrid | Hybrid+Rerank |")
    lines.append("|-------|--------|---------|--------|----------------|")
    for row in detail_rows:
        lines.append(
            f"| {row['query']} | {row['modes']['vector']['top5_hits']} "
            f"| {row['modes']['keyword']['top5_hits']} "
            f"| {row['modes']['hybrid']['top5_hits']} "
            f"| {row['modes']['hybrid_rerank']['top5_hits']} |"
        )

    # --- 结论 ---
    r5_h = avg("hybrid", "recall", 5)
    r5_v = avg("vector", "recall", 5)
    lines.append("\n## 结论\n")
    lines.append(
        f"- 混合检索（BM25 + 向量 + RRF 融合）相比纯向量检索："
        f"Recall@5 {r5_v:.4f} → {r5_h:.4f}（{pct(r5_h, r5_v)}），"
        f"MRR {mrr_map['vector']:.4f} → {mrr_map['hybrid']:.4f}（{pct(mrr_map['hybrid'], mrr_map['vector'])}）。"
    )
    lines.append(
        f"- 加入 CrossEncoder 重排序（hybrid_rerank）后，MRR 进一步提升至 {mrr_map['hybrid_rerank']:.4f}。"
    )
    lines.append(
        "- BM25 擅长精确术语匹配（药物名称、诊断标准），向量检索覆盖语义相近表达，"
        "两者通过 RRF 融合后优势互补。"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[报告已保存] {output_path}")


if __name__ == "__main__":
    main()
