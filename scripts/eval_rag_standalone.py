# -*- coding: utf-8 -*-
"""
RAG 检索召回准确率评估（自包含版，无外部依赖）

直接在内存中构建知识库 → 注入测试文档 → 运行评估
绕过 Milvus / sentence-transformers / SSL 等环境依赖，
但检索算法（BM25 Okapi / RRF / 向量余弦相似度）完全一致。

用法:
    source /d/Dev_tools/anaconda/etc/profile.d/conda.sh && conda activate Myagent
    python scripts/eval_rag_standalone.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ==============================================================================
# 嵌入 & 分词（从 rag_service.py 复刻，无 sentence-transformers 依赖）
# ==============================================================================

def _tokenize(text: str) -> list[str]:
    """中英文混合分词（jieba 中文分词 + 正则英文）。"""
    try:
        import jieba
        tokens = []
        # jieba 分词保留英文/数字作为独立 token
        for token in jieba.lcut(text):
            token = token.strip().lower()
            if not token:
                continue
            # 英文部分进一步按正则拆开
            if re.search(r"[a-z0-9]", token):
                tokens.extend([t for t in re.findall(r"[a-z0-9一-鿿]+", token) if t])
            else:
                tokens.append(token)
        return tokens
    except ImportError:
        return [t.lower() for t in re.findall(r"[\w一-鿿]+", text) if t]


class NGramEmbedding:
    """字符 n-gram 哈希嵌入（与项目 _NGramEmbedding 完全一致）。"""

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
                    # crc32 而非内置 hash()：内置 hash 受 PYTHONHASHSEED 影响，
                    # 跨进程不确定，会导致 CI smoke 门禁每次跑出不同的 Recall/MRR。
                    h = zlib.crc32(text[i:i + n].encode("utf-8")) % self._dim
                    vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ==============================================================================
# BM25 Okapi（与项目 _BM25 完全一致）
# ==============================================================================

class BM25:
    """内存 BM25 (Okapi) 关键词索引。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}
        self._N: int = 0
        self._idf: dict[str, float] = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        tf: dict[str, int] = {}
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
        self._idf = {}
        for t, df in self._df.items():
            self._idf[t] = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        if self._N == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: list[tuple[str, float, str]] = []
        k1, b = self.k1, self.b
        avgdl = self._avgdl or 1.0
        for i in range(self._N):
            tf = self._doc_freqs[i]
            dl = self._doc_lens[i]
            s = 0.0
            for t in q_terms:
                idf = self._idf.get(t, 0.0)
                if idf == 0.0:
                    continue
                f = tf.get(t, 0)
                if f == 0:
                    continue
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scores.append((self._ids[i], s, self._contents[i]))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ==============================================================================
# 内存文档存储（与项目 _InMemoryDocStore 完全一致）
# ==============================================================================

@dataclass
class RetrievalResult:
    id: str
    content: str
    score: float = 0.0
    source: str = ""


class InMemoryDocStore:
    """内存文档库：向量检索 + BM25 + 混合 (RRF)。"""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
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
        scored = []
        for d in self._docs:
            dot = sum(a * b for a, b in zip(query_vec, d["embedding"]))
            scored.append((dot, d))
        scored.sort(key=lambda x: -x[0])
        return [
            RetrievalResult(id=d["id"], content=d["content"], score=s, source="vector")
            for i, (s, d) in enumerate(scored[:top_k])
        ]

    def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        ranked = self._bm25.search(query, top_k)
        return [
            RetrievalResult(id=did, content=content, score=s, source="keyword")
            for did, s, content in ranked
        ]

    def hybrid_search(self, query_vec: list[float], query_text: str, top_k: int) -> list[RetrievalResult]:
        v = self.vector_search(query_vec, top_k * 2)
        k = self.keyword_search(query_text, top_k * 2)
        rrf_k = 60
        scores: dict[str, float] = {}
        best: dict[str, RetrievalResult] = {}
        for rank, r in enumerate(v):
            scores[r.id] = 1.0 / (rrf_k + rank + 1)
            best[r.id] = r
        for rank, r in enumerate(k):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if r.id not in best:
                best[r.id] = r
        merged_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            RetrievalResult(id=best[rid].id, content=best[rid].content,
                           score=scores[rid], source="hybrid")
            for rid in merged_ids
        ]


# ==============================================================================
# 评估指标
# ==============================================================================

def compute_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> dict[str, Any]:
    """计算单条 query 的所有指标。"""
    result: dict[str, Any] = {"recall": {}, "precision": {}, "mrr": 0.0, "ndcg": {}}

    if not relevant_ids:
        return result

    for k in k_values:
        top_k_set = set(retrieved_ids[:k])
        hits = len(top_k_set & relevant_ids)
        result["recall"][k] = hits / len(relevant_ids)
        result["precision"][k] = hits / k

    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            result["mrr"] = 1.0 / rank
            break

    for k in k_values:
        dcg = 0.0
        for rank, rid in enumerate(retrieved_ids[:k], start=1):
            if rid in relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
        result["ndcg"][k] = dcg / idcg if idcg > 0 else 0.0

    return result


# ==============================================================================
# 测试文档（心血管慢病领域）
# ==============================================================================

TEST_DOCUMENTS = [
    # 文档 1：高血压指南
    """
高血压诊断标准与分级

根据《中国高血压防治指南》，高血压诊断标准为：
在未使用降压药物的情况下，非同日3次测量诊室血压，SBP≥140mmHg和/或DBP≥90mmHg。

血压分级：
- 正常血压：收缩压<120mmHg 且 舒张压<80mmHg
- 正常高值：收缩压120-139mmHg 和/或 舒张压80-89mmHg
- 1级高血压（轻度）：收缩压140-159mmHg 和/或 舒张压90-99mmHg
- 2级高血压（中度）：收缩压160-179mmHg 和/或 舒张压100-109mmHg
- 3级高血压（重度）：收缩压≥180mmHg 和/或 舒张压≥110mmHg
- 单纯收缩期高血压：收缩压≥140mmHg 且 舒张压<90mmHg

高血压治疗目标：一般患者血压控制目标为<140/90mmHg，合并糖尿病、肾病、冠心病者目标为<130/80mmHg。
生活方式干预包括：限盐（每日<6g）、减重、规律运动、戒烟限酒。

常用降压药物包括：钙通道阻滞剂(CCB)、血管紧张素转换酶抑制剂(ACEI)、
血管紧张素受体拮抗剂(ARB)、利尿剂、β受体阻滞剂。
联合用药方案推荐：CCB+ACEI/ARB、ACEI/ARB+利尿剂、CCB+利尿剂等。
""",

    # 文档 2：冠心病
    """
冠心病的诊断与治疗

冠心病（Coronary Artery Disease, CAD）是冠状动脉粥样硬化导致血管狭窄或阻塞，
引起心肌缺血缺氧的心脏病。

常见症状包括：
- 胸痛（心绞痛）：胸骨后压榨性疼痛，可放射至左肩、左臂、下颌
- 胸闷、气短
- 心悸
- 严重者可出现心肌梗死：持续剧烈胸痛、大汗、濒死感

诊断方法：
- 心电图（ECG）：ST段压低或抬高、T波倒置
- 冠脉造影：诊断金标准，可明确狭窄部位和程度
- 运动负荷试验
- 冠脉CTA

治疗策略：
- 药物治疗：抗血小板（阿司匹林、氯吡格雷）、他汀类降脂药、β受体阻滞剂、硝酸酯类
- 介入治疗：PCI（经皮冠状动脉介入治疗）+ 支架植入
- 外科手术：冠脉搭桥术（CABG）

支架术后需双联抗血小板治疗（DAPT）：阿司匹林+氯吡格雷/替格瑞洛，
药物洗脱支架至少6-12个月，裸金属支架至少1个月。
""",

    # 文档 3：糖尿病
    """
2型糖尿病管理指南

诊断标准（满足任意一项）：
- 空腹血糖(FPG)≥7.0mmol/L
- OGTT 2小时血糖≥11.1mmol/L
- HbA1c≥6.5%
- 随机血糖≥11.1mmol/L且伴有典型高血糖症状

糖尿病并发症：
- 糖尿病肾病：分期依据eGFR和尿白蛋白。1期：eGFR≥90；2期：60-89；3期：30-59；4期：15-29；5期：<15
- 糖尿病视网膜病变
- 糖尿病周围神经病变
- 心血管疾病风险显著增加

治疗目标：
- HbA1c<7.0%（一般成人）
- 空腹血糖4.4-7.0mmol/L
- 餐后2h血糖<10.0mmol/L

预防措施：健康饮食、规律运动（每周≥150分钟中等强度）、控制体重、定期筛查。
""",

    # 文档 4：心力衰竭
    """
慢性心力衰竭诊治要点

心力衰竭是各种心脏疾病的终末阶段，主要表现为呼吸困难、乏力和液体潴留。

NYHA心功能分级：
- I级：日常活动不受限，一般活动不引起症状
- II级：体力活动轻度受限，休息时无症状，一般活动可出现症状
- III级：体力活动明显受限，小于一般活动即出现症状
- IV级：休息时也有症状，任何体力活动均加重不适

药物治疗：
- ACEI/ARB：所有HFrEF患者的基础用药
- β受体阻滞剂：比索洛尔、卡维地洛、美托洛尔缓释片
- 醛固酮受体拮抗剂：螺内酯
- ARNI（沙库巴曲缬沙坦）：替代ACEI/ARB首选
- SGLT2抑制剂：达格列净、恩格列净
- 利尿剂：呋塞米、氢氯噻嗪（改善症状）

治疗原则：黄金三角=ACEI/ARNI + β受体阻滞剂 + 醛固酮拮抗剂
""",

    # 文档 5：房颤与抗凝
    """
心房颤动抗凝治疗

房颤是临床最常见的心律失常，卒中风险显著增加。

抗凝指征评估（CHA2DS2-VASc评分）：
- 充血性心衰(1分)、高血压(1分)、年龄≥75岁(2分)、糖尿病(1分)
- 卒中/TIA史(2分)、血管疾病(1分)、年龄65-74岁(1分)、女性(1分)
- 男性≥2分、女性≥3分 推荐抗凝治疗

抗凝药物选择：
- 华法林：经典药物，需监测INR，目标2.0-3.0
- 新型口服抗凝药(NOAC)：利伐沙班、达比加群、阿哌沙班
- NOAC优势：无需常规监测、食物药物相互作用少、颅内出血风险更低

注意事项：
- 抗凝治疗需评估出血风险（HAS-BLED评分）
- 高出血风险并非抗凝禁忌，应积极纠正可逆因素
- 围手术期需根据肾功能调整停药和恢复时间
""",

    # 文档 6：血脂管理
    """
血脂异常临床管理

血脂检测项目：
- 总胆固醇(TC)
- 甘油三酯(TG)
- 低密度脂蛋白胆固醇(LDL-C)：主要致动脉粥样硬化因子
- 高密度脂蛋白胆固醇(HDL-C)：保护性因子

危险分层与LDL-C目标：
- 极高危：LDL-C<1.4mmol/L 且 降幅≥50%
- 高危：LDL-C<1.8mmol/L 且 降幅≥50%
- 中危：LDL-C<2.6mmol/L
- 低危：LDL-C<3.4mmol/L

极高危人群定义：ASCVD确诊者、糖尿病合并靶器官损害、严重CKD(eGFR<30)、
10年致死性CVD风险SCORE≥10%。

降脂药物：
- 他汀类（首选）：阿托伐他汀、瑞舒伐他汀。作用机制为抑制HMG-CoA还原酶，
  减少胆固醇合成，上调LDL受体
- 依折麦布：抑制肠道胆固醇吸收
- PCSK9抑制剂：适用于他汀不耐受或最大耐受量仍不达标者
""",
]


# ==============================================================================
# 文本分块
# ==============================================================================

class SimpleChunker:
    """简易分块器：按段落 + 窗口切分，近似 DocumentChunker(chunk_size=512, overlap=64)。"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """简单分块：先按双换行拆，再合并短段。"""
        text = text.strip()
        if not text:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) + 1 <= self.chunk_size:
                buf = (buf + "\n" + p).strip() if buf else p
            else:
                if buf:
                    chunks.append(buf)
                # 如果单段过长，继续按句号拆
                if len(p) > self.chunk_size:
                    sub = self._split_long(p)
                    chunks.extend(sub)
                else:
                    buf = p
        if buf:
            chunks.append(buf)
        return chunks or [text]

    def _split_long(self, text: str) -> list[str]:
        """长段按句号强制拆开。"""
        sentences = [s.strip() for s in text.replace("\n", " ").split("。") if s.strip()]
        result: list[str] = []
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= self.chunk_size:
                buf = (buf + "。" + s) if buf else s
            else:
                if buf:
                    result.append(buf + "。")
                buf = s
        if buf:
            result.append(buf + "。")
        return result or [text]


# ==============================================================================
# 主评估流程
# ==============================================================================

TEST_QUERIES = [
    {"query": "高血压的诊断标准是什么", "keywords": ["高血压", "诊断", "140", "90", "mmHg", "收缩压", "舒张压"]},
    {"query": "冠心病有哪些常见症状", "keywords": ["冠心病", "症状", "胸痛", "胸闷", "心绞痛", "放射"]},
    {"query": "如何预防2型糖尿病", "keywords": ["糖尿病", "预防", "血糖", "饮食", "运动", "筛查"]},
    {"query": "他汀类药物的作用机制", "keywords": ["他汀", "HMG-CoA", "胆固醇", "还原酶", "降脂"]},
    {"query": "房颤患者如何进行抗凝治疗", "keywords": ["房颤", "抗凝", "华法林", "利伐沙班", "INR", "NOAC"]},
    {"query": "心衰的NYHA分级标准", "keywords": ["心衰", "NYHA", "分级", "I级", "II级", "III级", "IV级"]},
    {"query": "高血压联合用药方案有哪些", "keywords": ["高血压", "联合用药", "CCB", "ACEI", "ARB", "利尿剂"]},
    {"query": "心肌梗死的急救措施", "keywords": ["心肌梗死", "PCI", "溶栓", "阿司匹林", "支架", "急救"]},
    {"query": "糖尿病肾病分期标准", "keywords": ["糖尿病", "肾病", "分期", "eGFR", "尿白蛋白", "1期", "2期"]},
    {"query": "血脂LDL-C控制目标是多少", "keywords": ["LDL-C", "目标", "1.4", "1.8", "极高危", "高危", "降幅"]},
    {"query": "支架植入术后用药注意事项", "keywords": ["支架", "术后", "DAPT", "双联抗血小板", "氯吡格雷"]},
    {"query": "心衰的药物治疗黄金三角是什么", "keywords": ["心衰", "黄金三角", "ACEI", "ARNI", "β受体阻滞剂", "醛固酮"]},
    {"query": "高血压生活方式干预包括哪些", "keywords": ["高血压", "生活方式", "限盐", "运动", "减重", "戒烟", "限酒"]},
    {"query": "冠脉造影的临床意义", "keywords": ["冠脉造影", "诊断", "金标准", "狭窄", "介入", "造影"]},
    {"query": "CHA2DS2-VASc评分有哪些项目", "keywords": ["CHA2DS2", "评分", "房颤", "年龄", "糖尿病", "卒中"]},
    {"query": "血压正常高值的范围是多少", "keywords": ["正常高值", "血压", "120", "139", "80", "89", "mmHg"]},
    {"query": "SGLT2抑制剂有哪些代表药物", "keywords": ["SGLT2", "达格列净", "恩格列净", "心衰", "糖尿病"]},
    {"query": "房颤抗凝出血风险评估用什么评分", "keywords": ["房颤", "出血", "HAS-BLED", "风险", "抗凝"]},
    {"query": "PCSK9抑制剂适用于哪些人群", "keywords": ["PCSK9", "他汀不耐受", "降脂", "LDL", "极高危"]},
    {"query": "冠脉搭桥术的英文缩写是什么", "keywords": ["CABG", "搭桥", "冠脉", "外科", "手术"]},
]


def auto_annotate(
    query: str, keywords: list[str], chunk_id: str, content: str, threshold: int = 2
) -> bool:
    """关键词匹配判定相关性：至少命中 threshold 个关键词。"""
    kw_hits = sum(1 for kw in keywords if kw.lower() in content.lower())
    return kw_hits >= min(threshold, max(1, len(keywords)))


def main() -> None:
    print("=" * 70)
    print("RAG 检索召回准确率评估（自包含版）")
    print("=" * 70)

    # 1. 构建知识库
    print("\n[1/4] 摄入测试文档...")
    embed_dim = 256
    embedder = NGramEmbedding(dim=embed_dim)
    chunker = SimpleChunker(chunk_size=300, chunk_overlap=50)
    store = InMemoryDocStore()

    chunk_id = 0
    doc_chunks: dict[str, str] = {}  # chunk_id → content
    for doc_idx, doc_text in enumerate(TEST_DOCUMENTS):
        chunks = chunker.chunk(doc_text)
        for chunk_text in chunks:
            cid = f"chunk_{chunk_id}"
            vec = embedder.encode(chunk_text)
            store.add(cid, chunk_text, vec)
            doc_chunks[cid] = chunk_text
            chunk_id += 1

    print(f"   文档数: {len(TEST_DOCUMENTS)}")
    print(f"   总块数: {store.count}")
    print(f"   嵌入维度: {embed_dim} (n-gram 哈希)")

    # 2. 自动标注 ground truth
    print("\n[2/4] 自动标注 ground truth...")
    query_annotations: list[dict[str, Any]] = []
    for tq in TEST_QUERIES:
        relevant_ids: set[str] = set()
        for cid, content in doc_chunks.items():
            if auto_annotate(tq["query"], tq["keywords"], cid, content, threshold=2):
                relevant_ids.add(cid)
        query_annotations.append({
            "query": tq["query"],
            "keywords": tq["keywords"],
            "relevant_ids": relevant_ids,
        })
        kw_preview = ", ".join(tq["keywords"][:4])
        print(f"   {tq['query'][:45]}... → {len(relevant_ids)} relevant chunks (kw: {kw_preview}...)")

    total_rel = sum(len(qa["relevant_ids"]) for qa in query_annotations)
    if total_rel == 0:
        print("\n[错误] 没有标注到任何相关 chunk！请检查关键词与文档内容的匹配度。")
        sys.exit(1)
    print(f"   总计标注相关 chunk: {total_rel} (平均 {total_rel/len(query_annotations):.1f}/query)")

    # 3. 运行评估
    print("\n[3/4] 运行三种检索模式评估...")
    k_values = [1, 3, 5, 10]
    modes = ["vector", "keyword", "hybrid"]
    all_metrics: dict[str, list[dict]] = {m: [] for m in modes}
    all_latency: dict[str, list[float]] = {m: [] for m in modes}

    for idx, qa in enumerate(query_annotations):
        query = qa["query"]
        relevant = qa["relevant_ids"]
        query_vec = embedder.encode(query)

        for mode in modes:
            t0 = time.perf_counter()
            if mode == "vector":
                results = store.vector_search(query_vec, top_k=max(k_values))
            elif mode == "keyword":
                results = store.keyword_search(query, top_k=max(k_values))
            else:  # hybrid
                results = store.hybrid_search(query_vec, query, top_k=max(k_values))
            elapsed = (time.perf_counter() - t0) * 1000

            retrieved_ids = [r.id for r in results]
            metrics = compute_metrics(retrieved_ids, relevant, k_values)
            metrics["elapsed_ms"] = elapsed
            metrics["query"] = query
            metrics["top5_hits"] = len(set(retrieved_ids[:5]) & relevant)
            all_metrics[mode].append(metrics)
            all_latency[mode].append(elapsed)

        if (idx + 1) % 5 == 0:
            print(f"   进度: {idx+1}/{len(query_annotations)}")

    # 4. 汇总结果
    print("\n[4/4] 汇总评估结果...\n")
    print("=" * 70)
    print("评估报告")
    print("=" * 70)

    # --- 汇总表 ---
    for metric_name in ["recall", "precision", "ndcg"]:
        print(f"\n{'─' * 50}")
        print(f"【{metric_name.upper()}@K】（越高越好）")
        header = f"{'Mode':<12}" + "".join(f"K={k:<8}" for k in k_values)
        print(header)
        print("-" * len(header))
        for mode in modes:
            avg_vals = []
            for k in k_values:
                vals = [m[metric_name].get(k, 0.0) for m in all_metrics[mode]]
                avg_vals.append(sum(vals) / len(vals))
            row = f"{mode:<12}" + "".join(f"{v:<8.4f}" for v in avg_vals)
            print(row)

    # --- MRR ---
    print(f"\n{'─' * 50}")
    print("【MRR — 平均倒数排名】（越高越好）")
    for mode in modes:
        avg_mrr = sum(m["mrr"] for m in all_metrics[mode]) / len(all_metrics[mode])
        print(f"  {mode:<12} {avg_mrr:.4f}")

    # --- 耗时 ---
    print(f"\n{'─' * 50}")
    print("【平均耗时 (ms)】")
    for mode in modes:
        avg_ms = sum(all_latency[mode]) / len(all_latency[mode])
        print(f"  {mode:<12} {avg_ms:.2f} ms")

    # --- 混合 vs 纯向量 提升 ---
    print(f"\n{'─' * 50}")
    print("【混合检索 vs 纯向量检索 — 提升幅度】")
    print("（正值表示 hybrid 更优）")
    for metric_name in ["recall", "precision", "ndcg"]:
        for k in k_values:
            v_h = sum(m[metric_name].get(k, 0.0) for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
            v_v = sum(m[metric_name].get(k, 0.0) for m in all_metrics["vector"]) / len(all_metrics["vector"])
            pct = (v_h - v_v) / v_v * 100 if v_v > 0 else 0.0
            print(f"  {metric_name.capitalize()}@{k:<4}: {pct:+.2f}%  (hybrid={v_h:.4f} vs vector={v_v:.4f})")

    h_mrr = sum(m["mrr"] for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
    v_mrr = sum(m["mrr"] for m in all_metrics["vector"]) / len(all_metrics["vector"])
    mrr_pct = (h_mrr - v_mrr) / v_mrr * 100 if v_mrr > 0 else 0.0
    print(f"  MRR           : {mrr_pct:+.2f}%  (hybrid={h_mrr:.4f} vs vector={v_mrr:.4f})")

    # --- 逐 query 详情 ---
    print(f"\n{'─' * 50}")
    print("【逐查询 Top-5 命中数】")
    print(f"{'Query':<45} {'Vec':>5} {'KW':>5} {'Hyb':>5}")
    print("-" * 65)
    for idx, qa in enumerate(query_annotations):
        v_hits = all_metrics["vector"][idx]["top5_hits"]
        k_hits = all_metrics["keyword"][idx]["top5_hits"]
        h_hits = all_metrics["hybrid"][idx]["top5_hits"]
        q_display = qa["query"][:42] + "..." if len(qa["query"]) > 42 else qa["query"]
        print(f"{q_display:<45} {v_hits:>5} {k_hits:>5} {h_hits:>5}")

    # --- 结论 ---
    print(f"\n{'=' * 70}")
    print("简历可引用结论：")
    print(f"{'=' * 70}")

    # 找 Recall@5 的提升
    r5_hyb = sum(m["recall"].get(5, 0.0) for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
    r5_vec = sum(m["recall"].get(5, 0.0) for m in all_metrics["vector"]) / len(all_metrics["vector"])
    r5_kw = sum(m["recall"].get(5, 0.0) for m in all_metrics["keyword"]) / len(all_metrics["keyword"])
    r5_gain = (r5_hyb - r5_vec) / r5_vec * 100 if r5_vec > 0 else 0.0

    p5_hyb = sum(m["precision"].get(5, 0.0) for m in all_metrics["hybrid"]) / len(all_metrics["hybrid"])
    p5_vec = sum(m["precision"].get(5, 0.0) for m in all_metrics["vector"]) / len(all_metrics["vector"])

    avg_hyb_ms = sum(all_latency["hybrid"]) / len(all_latency["hybrid"])

    print(f"\n混合检索（BM25 + 向量 + RRF 重排序）相比纯向量检索：")
    print(f"  Recall@5  从 {r5_vec:.4f} 提升至 {r5_hyb:.4f}（{r5_gain:+.2f}%）")
    print(f"  MRR       从 {v_mrr:.4f} 提升至 {h_mrr:.4f}（{mrr_pct:+.2f}%）")
    print(f"  平均耗时   {avg_hyb_ms:.2f} ms")

    # 结论摘要（可直接贴简历）
    print(f"\n── 简历文本（可直接使用）──")
    # 找合适的表述
    if r5_gain > 5:
        gain_text = f"Recall@5 提升 {r5_gain:.0f}%"
    else:
        gain_text = f"Recall@5 达到 {r5_hyb:.2f}（纯向量 {r5_vec:.2f}）"

    print(f"在20条心血管慢病领域测试查询上，混合检索（BM25+向量+RRF重排序）")
    print(f"相比纯向量检索，{gain_text}，")
    print(f"MRR 从 {v_mrr:.4f} 提升至 {h_mrr:.4f}，")
    print(f"单次检索平均耗时 {avg_hyb_ms:.0f}ms。")

    print(f"\n{'=' * 70}")
    print("评估完成。")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
