# -*- coding: utf-8 -*-
"""RAG 评测共享模块：单条指标计算 + 统计检验。

- EvalMetrics / compute_all_metrics：单条 query 的 Recall@K / Precision@K / MRR / NDCG@K
- bootstrap_ci：对「逐 query 指标值」做 bootstrap 均值置信区间
- paired_bootstrap_ci：对「逐 query 指标差 Δ_i」做配对 bootstrap 置信区间
- wilcoxon_signed_rank：配对样本显著性检验（非参数，不假设正态）

统计约定（重要）：所有模式间比较都是**配对**的——同一批 query 分别跑两个模式，
逐 query 得到一对指标，检验/置信区间都基于「逐 query 的指标差 Δ_i」，
不是把两组结果当独立样本混合。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class EvalMetrics:
    """一条 query 的评估指标。"""
    query: str
    mode: str
    recall: dict[int, float] = field(default_factory=dict)     # K → Recall@K
    precision: dict[int, float] = field(default_factory=dict)  # K → Precision@K
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)       # K → NDCG@K


def compute_all_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
    rel_grades: dict[str, float] | None = None,
) -> EvalMetrics:
    """对单条查询计算全部评估指标。

    :param relevant_ids: 相关 chunk 集合（分级标注下 = grade > 0 的 id）
    :param rel_grades: 可选的分级相关度 {chunk_id: grade}，grade ∈ {1=部分相关, 2=高度相关}。
        为 None 时退化为二元标注（所有相关 chunk 增益 = 1.0）。分级只影响 NDCG 的
        增益权重；Recall/Precision/MRR 只关心「相关与否」，不受分级影响。
    """
    metrics = EvalMetrics(query="", mode="")

    if not relevant_ids:
        return metrics

    # 相关性增益：分级用 grade，二元用 1.0
    def _gain(rid: str) -> float:
        if rel_grades is not None:
            return rel_grades.get(rid, 0.0)
        return 1.0 if rid in relevant_ids else 0.0

    # --- Recall@K & Precision@K ---
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        metrics.recall[k] = hits / len(relevant_ids) if relevant_ids else 0.0
        metrics.precision[k] = hits / k if k > 0 else 0.0

    # --- MRR: 第一个相关结果的倒数排名 ---
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            metrics.mrr = 1.0 / rank
            break

    # --- NDCG@K（分级相关 → 高度相关的 chunk 排在前面加分更多）---
    # 理想排序：所有相关 chunk 按 grade 降序排列后的折损增益和
    if rel_grades is not None:
        ideal_gains = sorted(
            (_gain(rid) for rid in relevant_ids), reverse=True
        )
    else:
        ideal_gains = [1.0] * len(relevant_ids)

    for k in k_values:
        dcg = sum(
            _gain(rid) / math.log2(rank + 1)
            for rank, rid in enumerate(retrieved_ids[:k], start=1)
        )
        idcg = sum(
            g / math.log2(rank + 1)
            for rank, g in enumerate(ideal_gains[:k], start=1)
        )
        metrics.ndcg[k] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """对一组「逐 query 指标值」做 bootstrap 均值置信区间。

    有放回重采样 n_bootstrap 次，取 alpha/2 与 1-alpha/2 分位数。
    返回 (lower, upper, mean)。
    """
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_bootstrap)]
    hi = means[min(int((1 - alpha / 2) * n_bootstrap), n_bootstrap - 1)]
    return (lo, hi, _mean(values))


def paired_bootstrap_ci(
    diffs: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """对「逐 query 指标差 Δ_i」做配对 bootstrap 置信区间。

    diffs[i] = metric_mode2(i) - metric_mode1(i)，必须逐 query 配对。
    重采样对象是 Δ_i 向量本身，而非把两组当独立样本混合（这是配对的关键）。
    """
    return bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank 检验（双侧，配对样本）。

    输入：diffs = 同一批 query 在两个模式下逐 query 的指标差 Δ_i（配对）。

    流程：剔除 Δ_i==0 的结 → 按 |Δ_i| 排序赋秩（结取平均秩）→ 分正负秩和
          → 检验统计量 W=min(W+,W-) → 正态近似（含结校正）算双侧 p 值。

    返回 (W, p_value)。n 较小时正态近似偏保守，n>20 时足够实用；
    需要精确小样本分布时改用 scipy.stats.wilcoxon。
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return (0.0, 1.0)

    abs_d = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: abs_d[i])

    # 赋秩（1-based，结取平均秩）
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    w_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nonzero[i] < 0)
    w_stat = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    # 结校正（t 为每个结组的大小）
    tie_corr = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        t = j - i
        tie_corr += t ** 3 - t
        i = j
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
    if var <= 0:
        return (w_stat, 1.0)

    z = (w_stat - mean) / math.sqrt(var)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return (w_stat, p)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cohens_kappa(
    annotator_a: dict[str, int],
    annotator_b: dict[str, int],
    categories: list[int] | None = None,
) -> float:
    """Cohen's Kappa：两个标注者对同一批样本的类别标注一致性。

    :param annotator_a/b: {item_id: category}，两个标注者必须覆盖同一批 item。
    :param categories: 可选类别全集；缺省取两标注者类别的并集。
    :return: Kappa ∈ [-1, 1]。1=完全一致，0=与随机一致无差别，<0=比随机还差。
             约定解读：<0.6 一致性不足（标准不清需先对齐），≥0.6 可用，≥0.8 良好。

    用途：量化 LLM-as-judge 标注的可信度——让两个独立 judge（不同提示/不同模型）
    标同一批候选，Kappa 低说明标注标准模糊，评测数字本身不可信。
    """
    items = sorted(set(annotator_a) | set(annotator_b))
    if not items:
        return 0.0
    if categories is None:
        categories = sorted(
            set(list(annotator_a.values()) + list(annotator_b.values()))
        )
    if not categories:
        return 0.0

    n = len(items)
    po = sum(
        1 for i in items if annotator_a.get(i) == annotator_b.get(i)
    ) / n

    pe = 0.0
    for c in categories:
        pa = sum(1 for i in items if annotator_a.get(i) == c) / n
        pb = sum(1 for i in items if annotator_b.get(i) == c) / n
        pe += pa * pb

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


if __name__ == "__main__":
    # --- 自测：二元 NDCG 与旧公式一致 ---
    rel = {"a", "b", "c"}
    m1 = compute_all_metrics(["a", "b", "c", "x", "y"], rel, [1, 3, 5])
    assert abs(m1.recall[5] - 1.0) < 1e-9, m1.recall
    assert abs(m1.mrr - 1.0) < 1e-9, m1.mrr
    assert abs(m1.ndcg[5] - 1.0) < 1e-9, m1.ndcg  # 相关项全在前三 → 理想顺序 → NDCG=1

    # --- 自测：分级 NDCG 比二元更能区分排序 ---
    grades = {"a": 2, "b": 1, "c": 1}
    # 把 grade=2 的 a 排最后 → 分级 NDCG 应低于把 a 排最前
    good = compute_all_metrics(["a", "b", "c", "x", "y"], rel, [3], rel_grades=grades)
    bad = compute_all_metrics(["b", "c", "a", "x", "y"], rel, [3], rel_grades=grades)
    assert good.ndcg[3] > bad.ndcg[3], (good.ndcg[3], bad.ndcg[3])

    # --- 自测：Cohen's Kappa ---
    # 完全一致 → 1.0
    assert abs(cohens_kappa({"a": 1, "b": 2}, {"a": 1, "b": 2}) - 1.0) < 1e-9
    # 随机独立 → 接近 0
    assert abs(cohens_kappa({"a": 1, "b": 2}, {"a": 2, "b": 1}) - (-1.0)) < 1e-9
    print("eval_metrics 自测通过：binary NDCG / graded NDCG / cohens_kappa")

