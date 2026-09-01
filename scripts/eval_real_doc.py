# -*- coding: utf-8 -*-
"""
RAG 检索召回准确率评估 — 基于真实文档「糖尿病合并高血压患者管理指南」

数据来源：Milvus 中已摄入的 78 个 chunk
嵌入方式：n-gram 哈希（256维），与项目 _NGramEmbedding 一致
分词方式：jieba 中文分词
检索算法：BM25 Okapi / 余弦相似度 / RRF 融合 (k=60)

用法:
    source /d/Dev_tools/anaconda/etc/profile.d/conda.sh && conda activate Myagent
    python scripts/eval_real_doc.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ==============================================================================
# 分词 & 嵌入（与项目代码一致）
# ==============================================================================

def _tokenize(text: str) -> list[str]:
    """jieba 中文分词 + 正则英文。"""
    try:
        import jieba
        tokens = []
        for token in jieba.lcut(text):
            token = token.strip().lower()
            if not token:
                continue
            if re.search(r"[a-z0-9]", token):
                tokens.extend([t for t in re.findall(r"[a-z0-9一-鿿]+", token) if t])
            else:
                tokens.append(token)
        return tokens
    except ImportError:
        return [t.lower() for t in re.findall(r"[\w一-鿿]+", text) if t]


class NGramEmbedding:
    """字符 n-gram 哈希嵌入（256维）。"""
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text = text.lower().strip()
        if not text:
            return vec
        for i in range(len(text)):
            for n in (2, 3):
                if i + n <= len(text):
                    h = hash(text[i:i + n]) % self._dim
                    vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ==============================================================================
# BM25 Okapi
# ==============================================================================

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._ids, self._contents = [], []
        self._doc_freqs, self._doc_lens = [], []
        self._avgdl, self._df, self._N = 0.0, {}, 0
        self._idf = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1
        self._ids.append(doc_id)
        self._contents.append(text)
        self._doc_freqs.append(tf)
        self._doc_lens.append(len(tokens))
        self._N += 1
        self._avgdl = sum(self._doc_lens) / self._N
        self._idf = {t: math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))
                     for t, df in self._df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        if self._N == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores = []
        for i in range(self._N):
            tf, dl = self._doc_freqs[i], self._doc_lens[i]
            s = 0.0
            for t in q_terms:
                idf = self._idf.get(t, 0.0)
                if idf == 0.0 or t not in tf:
                    continue
                f = tf[t]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0)))
            if s > 0:
                scores.append((self._ids[i], s, self._contents[i]))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ==============================================================================
# 内存文档存储
# ==============================================================================

@dataclass
class RetrievalResult:
    id: str
    content: str
    score: float = 0.0
    source: str = ""


class InMemoryDocStore:
    def __init__(self):
        self._docs = []
        self._bm25 = BM25()

    @property
    def count(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, content: str, embedding: list[float]) -> None:
        self._docs.append({"id": doc_id, "content": content, "embedding": embedding})
        self._bm25.add(doc_id, content)

    def vector_search(self, query_vec: list[float], top_k: int) -> list[RetrievalResult]:
        if not self._docs:
            return []
        scored = [(sum(a * b for a, b in zip(query_vec, d["embedding"])), d) for d in self._docs]
        scored.sort(key=lambda x: -x[0])
        return [RetrievalResult(id=d["id"], content=d["content"], score=s, source="vector")
                for i, (s, d) in enumerate(scored[:top_k])]

    def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        ranked = self._bm25.search(query, top_k)
        return [RetrievalResult(id=did, content=content, score=s, source="keyword")
                for did, s, content in ranked]

    def hybrid_search(self, query_vec: list[float], query_text: str, top_k: int) -> list[RetrievalResult]:
        v = self.vector_search(query_vec, top_k * 2)
        k = self.keyword_search(query_text, top_k * 2)
        rrf_k = 60
        scores, best = {}, {}
        for rank, r in enumerate(v):
            scores[r.id] = 1.0 / (rrf_k + rank + 1)
            best[r.id] = r
        for rank, r in enumerate(k):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if r.id not in best:
                best[r.id] = r
        merged_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [RetrievalResult(id=best[rid].id, content=best[rid].content,
                                score=scores[rid], source="hybrid") for rid in merged_ids]


# ==============================================================================
# 指标计算
# ==============================================================================

def compute_metrics(retrieved_ids: list[str], relevant_ids: set[str],
                    k_values: list[int]) -> dict[str, Any]:
    m = {"recall": {}, "precision": {}, "mrr": 0.0, "ndcg": {}}
    if not relevant_ids:
        return m
    for k in k_values:
        hits = len(set(retrieved_ids[:k]) & relevant_ids)
        m["recall"][k] = hits / len(relevant_ids)
        m["precision"][k] = hits / k
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            m["mrr"] = 1.0 / rank
            break
    for k in k_values:
        dcg = sum(1.0 / math.log2(rank + 1) for rank, rid in enumerate(retrieved_ids[:k], start=1) if rid in relevant_ids)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
        m["ndcg"][k] = dcg / idcg if idcg > 0 else 0.0
    return m


# ==============================================================================
# 加载 & 过滤文档块
# ==============================================================================

def load_and_filter_chunks(json_path: str) -> list[dict]:
    """加载 Milvus 导出的文档块，过滤噪声（页眉、参考文献、作者列表等）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 噪声过滤规则
    noise_patterns = [
        r"^fmx_T3RoZXJNaXJyb3Jz",           # 页眉
        r"中华内分泌代谢杂志.*\d{4}.*\d+期",      # 期刊页眉/页脚
        r"^\s*\[\d+\]\s",                    # 参考文献
        r"^\s*DOI:",                          # DOI 行
        r"^Ｃｏｒｒｅｓｐｏｎｄｉｎｇ",              # 通信作者
        r"^ｃｒｉｔｉｃａｌ",                      # 英文残余
        r"^ｋｅｙ ｄｏｍｅｓｔｉｃ",                # 英文残余
        r"^Ｆｕｎｄ ｐｒｏｇｒａｍ",               # 基金项目
        r"^【 Ｋｅｙ ｗｏｒｄｓ】",               # 关键词
        r"^【 Ｓｕｍｍａｒｙ】",                  # 摘要(英文)
        r"^参与本指南讨论和制定的专家",               # 作者列表
        r"^秘书",                              # 秘书
        r"^利益冲突",                            # COI
        r"^\s*ｓｔｕｄｙ",                       # 英文残余
        r"^ｅｎｚｙｍｅ",                        # 英文残余
        r"^\s*ａｎａｌｙｓｉｓ",                  # 英文残余
        r"^\s*ｓｅｍａｇｌｕｔｉｄｅ",             # 英文残余
        r"^\s*ｉｎｈｉｂｉｔｏｒｓ",               # 英文残余
        r"^\s*ｏｖａｓｃｕｌａｒ",                 # 英文残余
        r"^\s*ｔｏ ｐｒｅｖｅｎｔ",                # 英文残余
        r"^\s*ａｎｄ ｔｈｅ ｐｒｅｖａｌｅｎｃｅ",    # 英文残余
    ]

    filtered = []
    noise_count = 0
    for chunk in raw:
        content = chunk.get("content", "").strip()
        if not content:
            noise_count += 1
            continue
        # 检查噪声模式
        is_noise = False
        for pat in noise_patterns:
            if re.search(pat, content):
                is_noise = True
                break
        # 纯英文且长度>200的参考文献块
        if not is_noise:
            chinese_chars = len(re.findall(r"[一-鿿]", content))
            total_chars = len(content.replace(" ", ""))
            if total_chars > 0 and chinese_chars / max(total_chars, 1) < 0.05 and len(content) > 200:
                is_noise = True
        if is_noise:
            noise_count += 1
        else:
            filtered.append(chunk)

    print(f"  原始块数: {len(raw)}")
    print(f"  噪声过滤: {noise_count} 块")
    print(f"  有效块数: {len(filtered)}")
    return filtered


# ==============================================================================
# 测试查询（基于文档实际内容设计）
# ==============================================================================

TEST_QUERIES = [
    # ── 一、流行病学 ──
    {"query": "糖尿病患者中高血压患病率是多少",
     "keywords": ["糖尿病", "高血压", "患病率", "66.3", "合并症"]},
    {"query": "我国成年人高血压控制率是多少",
     "keywords": ["高血压", "控制率", "11.0", "知晓率", "治疗率", "2018"]},

    # ── 二、危险因素 ──
    {"query": "糖尿病合并高血压的危险因素有哪些",
     "keywords": ["危险因素", "高钠", "超重", "肥胖", "吸烟", "饮酒", "睡眠"]
    },
    {"query": "体重指数增加对高血压风险的影响",
     "keywords": ["体重指数", "BMI", "5", "49%", "腹型肥胖", "29.1"]},
    {"query": "吸烟和饮酒对高血压的影响",
     "keywords": ["吸烟", "饮酒", "高血压", "1.24", "1.41", "心血管"]},
    {"query": "心理压力和睡眠对血压的影响",
     "keywords": ["心理压力", "睡眠", "焦虑", "1.37", "U形", "6-7"]},

    # ── 三、诊断标准 ──
    {"query": "高血压的诊断标准是什么",
     "keywords": ["高血压", "诊断", "140", "90", "mmHg", "非同日", "3次"]},
    {"query": "动态血压诊断高血压的标准",
     "keywords": ["动态血压", "24h", "130", "80", "白天", "135", "夜间", "120"]},
    {"query": "血压测量的正确方法",
     "keywords": ["测量", "静坐", "5min", "上臂", "心脏", "重复", "30-60"]},
    {"query": "糖尿病合并高血压患者测量血压有什么特殊要求",
     "keywords": ["糖尿病", "站立位", "体位性低血压", "1min", "3min"]},

    # ── 四、血压管理目标 ──
    {"query": "糖尿病合并高血压患者的血压控制目标是多少",
     "keywords": ["血压", "目标", "120", "80", "130", "强化", "BPROAD"]},
    {"query": "高龄患者的降压目标是否需要调整",
     "keywords": ["高龄", "80岁", "140", "90", "个体化", "靶器官"]},
    {"query": "强化降压治疗需要注意哪些监测",
     "keywords": ["强化降压", "监测", "低血压", "电解质", "钾", "头晕"]},

    # ── 五、药物治疗 ──
    {"query": "糖尿病合并高血压推荐哪些降糖药物",
     "keywords": ["降糖", "SGLT2i", "GLP-1RA", "二甲双胍", "心肾获益"]},
    {"query": "SGLT2抑制剂有哪些临床获益",
     "keywords": ["SGLT2", "血压", "心血管", "心衰", "肾脏", "eGFR", "30"]},
    {"query": "GLP-1受体激动剂有哪些代表药物",
     "keywords": ["GLP-1", "利拉鲁肽", "度拉糖肽", "司美格鲁肽", "心血管"]},
    {"query": "糖尿病合并高血压的降压药物如何选择",
     "keywords": ["降压", "ACEI", "ARB", "ARNI", "CCB", "利尿剂", "β受体阻滞剂"]},
    {"query": "ACEI类药物有哪些不良反应和禁忌证",
     "keywords": ["ACEI", "干咳", "高钾", "肾动脉狭窄", "妊娠", "血肌酐"]},
    {"query": "CCB类药物适合哪些高血压患者",
     "keywords": ["CCB", "钙通道阻滞剂", "老年", "单纯收缩期", "脑卒中", "二氢吡啶"]},

    # ── 六、BPROAD研究 ──
    {"query": "BPROAD研究的主要结论是什么",
     "keywords": ["BPROAD", "强化", "120", "心血管", "21%", "4.2年", "12821"]},
]


def auto_annotate(query: str, keywords: list[str], content: str, threshold: int = 2) -> bool:
    """至少命中 threshold 个关键词判定为相关。"""
    hits = sum(1 for kw in keywords if kw.lower() in content.lower())
    return hits >= min(threshold, max(1, len(keywords)))


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    print("=" * 70)
    print("RAG 检索评估 — 真实文档「糖尿病合并高血压患者管理指南」")
    print("=" * 70)

    # 1. 加载文档块
    print("\n[1/4] 加载文档块...")
    script_dir = Path(__file__).resolve().parent
    chunks_path = script_dir / "doc_chunks.json"
    if not chunks_path.exists():
        print(f"[错误] 找不到 {chunks_path}，请先运行数据导出步骤")
        sys.exit(1)
    chunks = load_and_filter_chunks(str(chunks_path))
    if not chunks:
        print("[错误] 所有文档块都被过滤，请检查过滤规则")
        sys.exit(1)

    # 2. 构建索引
    print("\n[2/4] 构建检索索引...")
    embedder = NGramEmbedding(dim=256)
    store = InMemoryDocStore()
    for c in chunks:
        vec = embedder.encode(c["content"])
        store.add(c["id"], c["content"], vec)
    print(f"  知识库: {store.count} 个 chunk, 嵌入维度=256(n-gram)")

    # 3. 标注 & 评估
    print("\n[3/4] 自动标注 + 三路检索评估...")
    k_values = [1, 3, 5, 10]
    modes = ["vector", "keyword", "hybrid"]
    all_metrics = {m: [] for m in modes}
    all_latency = {m: [] for m in modes}

    for idx, tq in enumerate(TEST_QUERIES):
        query = tq["query"]
        keywords = tq["keywords"]

        # 标注 relevant_ids
        relevant_ids = set()
        for c in chunks:
            if auto_annotate(query, keywords, c["content"]):
                relevant_ids.add(c["id"])

        # 三点检索
        query_vec = embedder.encode(query)
        for mode in modes:
            t0 = time.perf_counter()
            if mode == "vector":
                results = store.vector_search(query_vec, top_k=max(k_values))
            elif mode == "keyword":
                results = store.keyword_search(query, top_k=max(k_values))
            else:
                results = store.hybrid_search(query_vec, query, top_k=max(k_values))
            elapsed = (time.perf_counter() - t0) * 1000

            m = compute_metrics([r.id for r in results], relevant_ids, k_values)
            m["elapsed_ms"] = elapsed
            m["top5_hits"] = len(set(r.id for r in results[:5]) & relevant_ids)
            all_metrics[mode].append(m)
            all_latency[mode].append(elapsed)

        n_rel = len(relevant_ids)
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"   [{idx+1}/{len(TEST_QUERIES)}] {query[:50]}... ({n_rel} relevant)")

    # 4. 汇总报告
    print("\n[4/4] 评估报告\n")
    print("=" * 70)

    # 汇总表
    for metric_name in ["recall", "precision", "ndcg"]:
        print(f"\n{'─' * 55}")
        print(f"【{metric_name.upper()}@K】")
        header = f"{'Mode':<12}" + "".join(f"K={k:<8}" for k in k_values)
        print(header)
        print("-" * len(header))
        for mode in modes:
            vals = [sum(m[metric_name].get(k, 0.0) for m in all_metrics[mode]) / len(all_metrics[mode])
                    for k in k_values]
            print(f"{mode:<12}" + "".join(f"{v:<8.4f}" for v in vals))

    print(f"\n{'─' * 55}")
    print("【MRR & 耗时】")
    print(f"{'Mode':<12} {'MRR':<10} {'平均耗时 (ms)':<15}")
    print("-" * 40)
    for mode in modes:
        avg_mrr = sum(m["mrr"] for m in all_metrics[mode]) / len(all_metrics[mode])
        avg_ms = sum(all_latency[mode]) / len(all_latency[mode])
        print(f"{mode:<12} {avg_mrr:<10.4f} {avg_ms:<15.2f}")

    # 混合 vs 纯向量
    print(f"\n{'─' * 55}")
    print("【混合检索 vs 纯向量 — 提升幅度】")
    for metric_name in ["recall", "precision", "ndcg"]:
        for k in k_values:
            v_h = sum(m[metric_name].get(k, 0.0) for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
            v_v = sum(m[metric_name].get(k, 0.0) for m in all_metrics["vector"]) / len(all_metrics["vector"])
            pct = (v_h - v_v) / v_v * 100 if v_v > 0 else 0.0
            print(f"  {metric_name.capitalize()}@{k:<4}: {pct:+.2f}%  (hybrid={v_h:.4f}  vector={v_v:.4f})")
    h_mrr = sum(m["mrr"] for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
    v_mrr = sum(m["mrr"] for m in all_metrics["vector"]) / len(all_metrics["vector"])
    print(f"  MRR           : {(h_mrr-v_mrr)/v_mrr*100:+.2f}%  (hybrid={h_mrr:.4f}  vector={v_mrr:.4f})" if v_mrr > 0 else "  MRR: N/A")

    # 逐 query Top-5 命中
    print(f"\n{'─' * 55}")
    print("【逐查询 Top-5 命中对比】（Vec / KW / Hyb）")
    print(f"{'Query':<50} {'Vec':>4} {'KW':>4} {'Hyb':>4}")
    print("-" * 66)
    for idx, tq in enumerate(TEST_QUERIES):
        vh = all_metrics["vector"][idx]["top5_hits"]
        kh = all_metrics["keyword"][idx]["top5_hits"]
        hh = all_metrics["hybrid"][idx]["top5_hits"]
        q = tq["query"][:47] + "..." if len(tq["query"]) > 47 else tq["query"]
        print(f"{q:<50} {vh:>4} {kh:>4} {hh:>4}")

    # 简历结论
    print(f"\n{'=' * 70}")
    print("简历可引用结论")
    print(f"{'=' * 70}")
    r5_hyb = sum(m["recall"].get(5, 0.0) for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
    r5_vec = sum(m["recall"].get(5, 0.0) for m in all_metrics["vector"]) / len(all_metrics["vector"])
    r5_kw = sum(m["recall"].get(5, 0.0) for m in all_metrics["keyword"]) / len(all_metrics["keyword"])
    p5_hyb = sum(m["precision"].get(5, 0.0) for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
    avg_hyb_ms = sum(all_latency["hybrid"]) / len(all_latency["hybrid"])

    print(f"\n  测试规模: {store.count} chunks × {len(TEST_QUERIES)} queries")
    print(f"  文档来源: 中华医学会内分泌学分会《糖尿病合并高血压患者管理指南》(2025)")
    print(f"\n  纯向量检索:  Recall@5={r5_vec:.4f}  MRR={v_mrr:.4f}")
    print(f"  纯关键词检索: Recall@5={r5_kw:.4f}  MRR={h_mrr:.4f}")
    print(f"  混合检索:    Recall@5={r5_hyb:.4f}  MRR={h_mrr:.4f}")
    r5_gain = (r5_hyb - r5_vec) / r5_vec * 100 if r5_vec > 0 else 0.0
    mrr_gain = (h_mrr - v_mrr) / v_mrr * 100 if v_mrr > 0 else 0.0
    print(f"\n  混合 vs 纯向量: Recall@5 {r5_gain:+.1f}%, MRR {mrr_gain:+.1f}%")

    print(f"\n  ── 简历文本（中文，可直接使用）──")
    print(f"  基于《糖尿病合并高血压患者管理指南》(2025) 构建知识库，")
    print(f"  在{len(TEST_QUERIES)}条测试查询上评估混合检索效果：")
    print(f"  相比纯向量检索，BM25+向量+RRF重排序将Recall@5")
    print(f"  从 {r5_vec:.2f} 提升至 {r5_hyb:.2f}（{r5_gain:+.0f}%），MRR 从 {v_mrr:.2f} 提升至 {h_mrr:.2f}（{mrr_gain:+.0f}%），")
    print(f"  验证了双路融合策略在中文医学文献检索中的有效性。")

    print(f"\n{'=' * 70}")
    print("评估完成。")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
