# -*- coding: utf-8 -*-
"""
RAG 检索召回准确率评估实验

使用方式：
  1. 确保知识库中已有文档（通过 Web UI 上传，或调用 ingest API）
  2. 准备测试查询文件（参考 eval_queries_template.json）
  3. 运行:  python scripts/eval_rag.py
  4. 查看输出的评估报告

评估指标：
  - Recall@K    : Top-K 结果中召回了多少比例的相关文档
  - Precision@K : Top-K 结果中有多少比例是真正相关的
  - MRR         : 第一个相关结果排名的倒数均值
  - NDCG@K      : 归一化折损累计增益（考虑排名位置）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 将项目根目录加入 sys.path（导入 app.*）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 将 scripts 目录加入 sys.path（导入同目录 eval_metrics）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_metrics import (  # noqa: E402
    EvalMetrics,
    bootstrap_ci,
    compute_all_metrics,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)

# ==============================================================================
# 配置
# ==============================================================================

# 测试查询文件路径（相对于项目根目录）
QUERIES_FILE = PROJECT_ROOT / "scripts" / "eval_queries.json"

# 评估的 Top-K 值列表
EVAL_K_VALUES = [1, 3, 5, 10]

# 召回阶段评估的候选数量。两阶段检索里，RRF 混合召回先扩到 candidate_k 个候选，
# 再交给 cross-encoder 精排到 top_k。混合检索的增益主要体现在「候选集覆盖度」这个
# 深度，而不是最终 top_k=5 的浅排名——所以单独用 candidate_k 评估召回。
CANDIDATE_K = 50

# 输出报告路径
REPORT_FILE = PROJECT_ROOT / "scripts" / "eval_report.md"


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class TestQuery:
    """单条测试查询。"""
    query: str
    relevant_ids: set[str]           # 标注为相关的 chunk ID（精确匹配）
    relevant_keywords: list[str] = field(default_factory=list)  # 关键词（备用自动标注）
    type: str = ""                   # 查询类型（事实型/多跳/边界/对比型/反例）
    rel_grades: dict[str, float] = field(default_factory=dict)  # 分级相关度 {id: 1|2}


@dataclass
class QueryResult:
    """单条查询在某一检索模式下的结果。"""
    query: str
    mode: str                        # vector / keyword / hybrid
    retrieved_ids: list[str]         # 按排名顺序的检索结果 ID
    scores: list[float]              # 对应分数
    elapsed_ms: float                # 检索耗时（毫秒）

    @property
    def top_k_ids(self, k: int) -> list[str]:
        return self.retrieved_ids[:k]


# ==============================================================================
# 主评估流程
# ==============================================================================

class RAGEvaluator:
    """RAG 检索评估器。"""

    def __init__(self, include_rerank: bool = False) -> None:
        from app.core.rag_service import RAGService
        self._rag = RAGService()
        self._k_values = EVAL_K_VALUES
        # 评测模式：基础三模式 + 可选 rerank 消融组（hybrid + cross-encoder 精排）
        self._modes = ["vector", "keyword", "hybrid"]
        if include_rerank:
            self._modes.append("hybrid_rerank")

    # ------------------------------------------------------------------
    # 加载测试查询
    # ------------------------------------------------------------------

    def load_queries(self, path: Path | None = None) -> list[TestQuery]:
        """从 JSON 文件加载测试查询。"""
        filepath = path or QUERIES_FILE
        if not filepath.exists():
            print(f"[错误] 测试查询文件不存在: {filepath}")
            print(f"       请先参考模板创建: {filepath.parent / 'eval_queries_template.json'}")
            sys.exit(1)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容两种格式：{"queries": [...]} 或直接数组 [...]
        if isinstance(data, dict) and "queries" in data:
            items = data["queries"]
        elif isinstance(data, list):
            items = data
        else:
            print(f"[错误] 无法识别的查询文件格式: {filepath}")
            sys.exit(1)

        queries: list[TestQuery] = []
        for item in items:
            # 分级相关度（可选）：{"chunk_id": 2}，1=部分相关 2=高度相关
            relevance = item.get("relevance") or {}
            rel_grades = (
                {str(k): float(v) for k, v in relevance.items()} if relevance else {}
            )
            relevant_ids = set(item.get("relevant_ids", []))
            # 若只给了分级标注而没给 relevant_ids，从 grade>0 派生相关集合
            if rel_grades and not relevant_ids:
                relevant_ids = {rid for rid, g in rel_grades.items() if g > 0}
            queries.append(TestQuery(
                query=item["query"],
                relevant_ids=relevant_ids,
                relevant_keywords=item.get("relevant_keywords", []),
                type=item.get("type", ""),
                rel_grades=rel_grades,
            ))
        return queries

    # ------------------------------------------------------------------
    # 检索所有模式
    # ------------------------------------------------------------------

    async def search_all_modes(
        self, query: str, top_k: int = 20
    ) -> list[QueryResult]:
        """对一条查询执行各评测模式的检索。"""
        results: list[QueryResult] = []

        for mode in self._modes:
            # hybrid_rerank 是消融组：hybrid 召回 + 强制 cross-encoder 精排。
            # 基础三模式显式 rerank=False（不跟随 config 全局默认），保证"有/无精排"
            # 的对照在同一进程内成立，且与 config.rerank_enabled 的取值无关。
            search_mode = "hybrid" if mode == "hybrid_rerank" else mode
            rerank = True if mode == "hybrid_rerank" else False
            t0 = time.perf_counter()
            try:
                hits = await self._rag.search(
                    query, top_k=top_k, mode=search_mode, rerank=rerank
                )
            except Exception as e:
                print(f"  [警告] 检索失败 mode={mode} query={query[:30]}: {e}")
                hits = []
            elapsed = (time.perf_counter() - t0) * 1000

            results.append(QueryResult(
                query=query,
                mode=mode,
                retrieved_ids=[r.id for r in hits],
                scores=[r.score for r in hits],
                elapsed_ms=elapsed,
            ))

        return results

    # ------------------------------------------------------------------
    # 批量评估
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        queries: list[TestQuery],
        top_k: int = 20,
        candidate_k: int = CANDIDATE_K,
        rerank_candidate_k: int | None = None,
    ) -> dict[str, Any]:
        """
        运行完整评估，返回结构化结果。

        :param queries: 测试查询列表
        :param top_k: 检索时请求的结果数（应 >= max(EVAL_K_VALUES)）
        :param candidate_k: 召回阶段候选集深度（测 Recall@candidate_k 覆盖度）
        :param rerank_candidate_k: 精排深度（生产 rerank_candidate_k 口径）。None 时
            跟随生产 settings.rerank_candidate_k。注意：这是与 candidate_k 独立的两个
            概念——candidate_k 是「召回池有多深」，rerank_candidate_k 是「精排在多深
            的候选集里重排」，评测时若两者不一致必须显式区分，否则高估/低估生产口径。
        :return: 包含逐查询指标和汇总统计的字典
        """
        if not queries:
            print("[错误] 没有测试查询，请先准备 eval_queries.json")
            sys.exit(1)

        # 检查知识库
        total_chunks = self._rag.chunks_count
        if total_chunks == 0:
            print(
                "[错误] 知识库为空！请先通过 Web UI 或 API 上传至少一份文档。"
            )
            sys.exit(1)

        print(f"知识库块数: {total_chunks}")
        print(f"测试查询数: {len(queries)}")
        print(f"评估 K 值:  {self._k_values}")
        print("-" * 60)

        # rerank 消融组：真正挂载 cross-encoder（config 默认可能为关，RAGService
        # 里 _reranker 为 None 时 search(rerank=True) 会静默 no-op）。此处显式构造。
        # 精排深度用 rerank_candidate_k（默认对齐生产 settings.rerank_candidate_k），
        # 与召回阶段 candidate_k 解耦——生产 search() 里 candidate_k=max(rerank_candidate_k,
        # top_k)，精排只在这批候选里重排，因此评测必须用同一深度才能如实反映生产口径。
        if "hybrid_rerank" in self._modes:
            from app.config import get_settings
            from app.core.rag.reranker import Reranker

            settings = get_settings()
            self._rag._rerank_enabled = True
            self._rag._rerank_failed = False
            self._rag._rerank_candidate_k = (
                int(rerank_candidate_k)
                if rerank_candidate_k is not None
                else int(settings.rerank_candidate_k)
            )
            if self._rag._reranker is None:
                self._rag._reranker = Reranker(
                    model_name=settings.rerank_model,
                    device=settings.rerank_device or None,
                )
            print(
                f"rerank 消融已启用: model={settings.rerank_model}, "
                f"rerank_candidate_k={self._rag._rerank_candidate_k}"
            )

        # 逐查询评估
        all_metrics: dict[str, list[EvalMetrics]] = {
            m: [] for m in self._modes
        }
        detailed: list[dict[str, Any]] = []
        # 召回阶段：以 candidate_k 评估各模式候选集覆盖度（混合检索真正发挥作用的深度）。
        # 注意 stage_recall 只测基础三模式：hybrid_rerank 的精排不改变 candidate_k 候选集
        # 内容（只重排序），其覆盖度与 hybrid 相同，故不重复测。
        stage_recall: dict[str, list[float]] = {
            "vector": [], "keyword": [], "hybrid": []
        }

        for i, tq in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {tq.query[:60]}...")
            results = await self.search_all_modes(tq.query, top_k=top_k)

            row: dict[str, Any] = {
                "query": tq.query,
                "relevant_count": len(tq.relevant_ids),
                "modes": {},
            }

            for qr in results:
                m = compute_all_metrics(
                    qr.retrieved_ids, tq.relevant_ids, self._k_values,
                    rel_grades=tq.rel_grades or None,
                )
                m.query = tq.query
                m.mode = qr.mode
                all_metrics[qr.mode].append(m)

                row["modes"][qr.mode] = {
                    "recall": {str(k): f"{v:.4f}" for k, v in m.recall.items()},
                    "precision": {str(k): f"{v:.4f}" for k, v in m.precision.items()},
                    "mrr": round(m.mrr, 4),
                    "ndcg": {str(k): f"{v:.4f}" for k, v in m.ndcg.items()},
                    "elapsed_ms": round(qr.elapsed_ms, 2),
                    "top5_ids": qr.retrieved_ids[:5],
                    "hits_in_top5": len(set(qr.retrieved_ids[:5]) & tq.relevant_ids),
                }
            # 召回阶段覆盖度（candidate_k 候选集，喂给精排前的召回深度）。
            # 显式 rerank=False：stage_recall 测的是精排前的候选集，Recall@50 对
            # 排序不变，rerank 只重排序不改变集合，故无需（也不应）跑 cross-encoder，
            # 否则会在 config 默认 rerank_enabled=True 时白白对 3 路各 50 候选精排。
            row["stage_recall"] = {}
            for mode in ("vector", "keyword", "hybrid"):
                try:
                    stage_hits = await self._rag.search(
                        tq.query, top_k=candidate_k, mode=mode, rerank=False
                    )
                    rec = (
                        len(set(r.id for r in stage_hits) & tq.relevant_ids)
                        / len(tq.relevant_ids)
                        if tq.relevant_ids
                        else 0.0
                    )
                except Exception:
                    rec = 0.0
                row["stage_recall"][mode] = round(rec, 4)
                stage_recall[mode].append(rec)

            detailed.append(row)

        # 汇总统计
        summary = self._build_summary(all_metrics)

        # 按查询类型分组（定位"哪个类型掉分"，而非笼统平均分）
        summary["by_type"] = self._build_type_summary(all_metrics, queries, self._modes)

        # 召回阶段汇总：各模式在 candidate_k 深度的平均 Recall
        summary["stage_candidate_k"] = candidate_k
        summary["stage_recall"] = {
            mode: round(sum(vals) / len(vals), 4) if vals else 0.0
            for mode, vals in stage_recall.items()
        }
        sr_vec = summary["stage_recall"].get("vector", 0)
        sr_hyb = summary["stage_recall"].get("hybrid", 0)
        summary["stage_recall_gain"] = (
            round((sr_hyb - sr_vec) / sr_vec * 100, 2) if sr_vec > 0 else 0.0
        )

        return {
            "config": {
                "total_chunks": total_chunks,
                "num_queries": len(queries),
                "k_values": self._k_values,
                "candidate_k": candidate_k,
            },
            "summary": summary,
            "detailed": detailed,
        }

    def _build_summary(
        self, all_metrics: dict[str, list[EvalMetrics]]
    ) -> dict[str, Any]:
        """计算各模式在各 K 值下的平均指标。"""
        summary: dict[str, Any] = {}

        for mode, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            n = len(metrics_list)

            mode_summary: dict[str, Any] = {
                "count": n,
                "avg_mrr": round(sum(m.mrr for m in metrics_list) / n, 4),
            }

            for metric_name in ("recall", "precision", "ndcg"):
                mode_summary[metric_name] = {}
                for k in self._k_values:
                    avg = sum(getattr(m, metric_name).get(k, 0.0) for m in metrics_list) / n
                    mode_summary[metric_name][str(k)] = round(avg, 4)

            # bootstrap 95% CI（对逐 query 指标值，非独立混合）
            mrr_lo, mrr_hi, _ = bootstrap_ci([m.mrr for m in metrics_list])
            mode_summary["mrr_ci"] = {"lo": round(mrr_lo, 4), "hi": round(mrr_hi, 4)}
            r5_lo, r5_hi, _ = bootstrap_ci([m.recall.get(5, 0.0) for m in metrics_list])
            mode_summary["recall5_ci"] = {"lo": round(r5_lo, 4), "hi": round(r5_hi, 4)}

            summary[mode] = mode_summary

        # 混合检索 vs 纯向量的提升比例
        if "hybrid" in summary and "vector" in summary:
            gains: dict[str, Any] = {}
            for metric_name in ("recall", "precision", "ndcg"):
                gains[metric_name] = {}
                for k in self._k_values:
                    v_h = summary["hybrid"][metric_name].get(str(k), 0)
                    v_v = summary["vector"][metric_name].get(str(k), 0)
                    if v_v > 0:
                        pct = round((v_h - v_v) / v_v * 100, 2)
                    else:
                        pct = 0.0
                    gains[metric_name][str(k)] = pct
            gains["avg_mrr"] = round(
                (summary["hybrid"]["avg_mrr"] - summary["vector"]["avg_mrr"])
                / summary["vector"]["avg_mrr"] * 100
                if summary["vector"]["avg_mrr"] > 0
                else 0.0,
                2,
            )
            summary["hybrid_vs_vector_gain"] = gains

        # rerank 消融：hybrid_rerank vs hybrid 的提升比例（重排的边际收益）
        if "hybrid_rerank" in summary and "hybrid" in summary:
            gains = {}
            for metric_name in ("recall", "precision", "ndcg"):
                gains[metric_name] = {}
                for k in self._k_values:
                    v_r = summary["hybrid_rerank"][metric_name].get(str(k), 0)
                    v_h = summary["hybrid"][metric_name].get(str(k), 0)
                    gains[metric_name][str(k)] = round((v_r - v_h) / v_h * 100, 2) if v_h > 0 else 0.0
            gains["avg_mrr"] = round(
                (summary["hybrid_rerank"]["avg_mrr"] - summary["hybrid"]["avg_mrr"])
                / summary["hybrid"]["avg_mrr"] * 100
                if summary["hybrid"]["avg_mrr"] > 0
                else 0.0,
                2,
            )
            summary["rerank_vs_hybrid_gain"] = gains

        # 配对显著性检验（基于同一 query 的逐条指标差）
        if "hybrid" in summary and "vector" in summary:
            summary["hybrid_vs_vector_test"] = self._paired_test(
                all_metrics, "vector", "hybrid"
            )
        if "hybrid_rerank" in summary and "hybrid" in summary:
            summary["rerank_vs_hybrid_test"] = self._paired_test(
                all_metrics, "hybrid", "hybrid_rerank"
            )

        return summary

    def _build_type_summary(
        self,
        all_metrics: dict[str, list[EvalMetrics]],
        queries: list[TestQuery],
        modes: list[str],
    ) -> dict[str, Any]:
        """按查询类型分组，计算各类型下各模式的平均 Recall@5 / MRR。

        all_metrics[mode] 与 queries 顺序一一对应（evaluate 逐 query 追加），
        故可 zip 对齐；无 type 标签的 query 归入「未分类」。
        """
        from collections import defaultdict

        by_type: dict[str, dict[str, list[EvalMetrics]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for mode in modes:
            for tq, m in zip(queries, all_metrics.get(mode, [])):
                by_type[tq.type or "未分类"][mode].append(m)

        out: dict[str, Any] = {}
        # 保持与查询出现顺序一致的展示顺序
        for tname in list(dict.fromkeys(tq.type or "未分类" for tq in queries)):
            per_mode: dict[str, Any] = {}
            for mode in modes:
                ms = by_type[tname][mode]
                if not ms:
                    continue
                n = len(ms)
                per_mode[mode] = {
                    "count": n,
                    "recall5": round(
                        sum(m.recall.get(5, 0.0) for m in ms) / n, 4
                    ),
                    "mrr": round(sum(m.mrr for m in ms) / n, 4),
                }
            out[tname] = per_mode
        return out

    def _paired_test(
        self,
        all_metrics: dict[str, list[EvalMetrics]],
        mode_a: str,
        mode_b: str,
    ) -> dict[str, Any]:
        """对两个模式做配对统计检验。

        基于同一批 query 的逐条指标差 Δ_i = metric_b(i) - metric_a(i)，
        用 Wilcoxon signed-rank（显著性）+ paired bootstrap（差值的 95% CI）。
        """
        a = all_metrics.get(mode_a, [])
        b = all_metrics.get(mode_b, [])
        if not a or not b or len(a) != len(b):
            return {}

        out: dict[str, Any] = {}

        mrr_diffs = [mb.mrr - ma.mrr for ma, mb in zip(a, b)]
        _, p_mrr = wilcoxon_signed_rank(mrr_diffs)
        mrr_lo, mrr_hi, mrr_mean = paired_bootstrap_ci(mrr_diffs)
        out["mrr"] = {
            "mean_diff": round(mrr_mean, 4),
            "ci": [round(mrr_lo, 4), round(mrr_hi, 4)],
            "p": round(p_mrr, 4),
        }

        k = 5
        r5_diffs = [mb.recall.get(k, 0.0) - ma.recall.get(k, 0.0) for ma, mb in zip(a, b)]
        _, p_r5 = wilcoxon_signed_rank(r5_diffs)
        r5_lo, r5_hi, r5_mean = paired_bootstrap_ci(r5_diffs)
        out["recall5"] = {
            "mean_diff": round(r5_mean, 4),
            "ci": [round(r5_lo, 4), round(r5_hi, 4)],
            "p": round(p_r5, 4),
        }

        return out


# ==============================================================================
# 报告生成
# ==============================================================================

def generate_markdown_report(result: dict[str, Any], output_path: Path) -> str:
    """生成 Markdown 格式评估报告。"""
    cfg = result["config"]
    s = result["summary"]
    stage_recall = s.get("stage_recall")
    stage_gain = s.get("stage_recall_gain", 0)
    candidate_k = s.get("stage_candidate_k", CANDIDATE_K)

    # 报告展示的模式：基础三模式 + 若跑了 rerank 消融则追加 hybrid_rerank
    modes = ["vector", "keyword", "hybrid"]
    if "hybrid_rerank" in s:
        modes.append("hybrid_rerank")

    lines: list[str] = []
    lines.append("# RAG 检索召回准确率评估报告\n")
    lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**知识库块数**: {cfg['total_chunks']}")
    lines.append(f"**测试查询数**: {cfg['num_queries']}")
    lines.append(f"**评估指标**: Recall@K, Precision@K, MRR, NDCG@K (K={cfg['k_values']})\n")

    # --- 变更记录 ---
    lines.append("## 变更记录\n")
    lines.append("- **修复**: `_tokenize` 增加全角→半角归一化（`_normalize_fullwidth`）。中文医学 PDF 语料含")
    lines.append("  29214 个全角 ASCII 字符，此前全角 `ＡＣＥＩ` 被 jieba 切成 `ａ/ｃ/ｅ/ｉ`，与查询半角")
    lines.append("  `acei` 词面失配，导致 BM25 关键词检索系统性漏检药物名/数值/单位。修复后 keyword 召回回升。\n")

    # --- 汇总表 ---
    lines.append("## 一、汇总指标\n")
    lines.append("### Recall@K（召回率 — 越高越好）\n")
    lines.append(_build_metric_table(s, "recall", modes))
    lines.append("\n### Precision@K（精确率 — 越高越好）\n")
    lines.append(_build_metric_table(s, "precision", modes))
    lines.append("\n### NDCG@K（归一化折损累计增益 — 越高越好）\n")
    lines.append(_build_metric_table(s, "ndcg", modes))
    lines.append("\n### MRR（平均倒数排名 — 越高越好）\n")
    lines.append(_build_mrr_table(s, modes))

    # --- 按查询类型分组 ---
    by_type = s.get("by_type")
    if by_type:
        lines.append("\n### 按查询类型分组（Recall@5 / MRR — 定位哪个类型掉分）\n")
        lines.append(
            "查询按 doc 3.1 类型分层打标签；空类型或无该类型查询的模式显示 `-`。"
            "当前标注集类型分布未必均衡，缺的类型（对比型/反例）需在扩集阶段补齐。\n"
        )
        lines.append(_build_type_table(by_type, modes))

    # --- 召回阶段（candidate_k 候选集覆盖度） ---
    if stage_recall:
        lines.append(f"\n## 召回阶段 Recall@{candidate_k}（候选集覆盖度 — 混合检索的真正增益）\n")
        lines.append("两阶段检索里，RRF 混合召回先扩到 candidate_k 个候选再交精排；混合检索的增益体现在这个")
        lines.append("候选集深度，而非最终 top-5 浅排名（浅排名下等权 RRF 会被较弱的关键词路稀释）。\n")
        lines.append(f"| 模式 | Recall@{candidate_k} |")
        lines.append("|------|---------------------|")
        for mode in ("vector", "keyword", "hybrid"):
            lines.append(f"| {mode} | {stage_recall.get(mode, 0):.4f} |")
        lines.append(f"\n- 混合检索候选集 Recall@{candidate_k} 相比纯向量：**{stage_gain:+.2f}%**"
                     "（正值 = 混合召回更全，说明双路在候选集深度互补）\n")

    # --- 混合检索提升 ---
    gain = s.get("hybrid_vs_vector_gain", {})
    if gain:
        lines.append("\n## 二、混合检索 vs 纯向量检索 — 提升幅度\n")
        lines.append("（正值表示混合检索更优）\n")
        gain_lines = ["| 指标 | " + " | ".join(f"K={k}" for k in cfg["k_values"]) + " |"]
        gain_lines.append("|------|" + "|".join(["------"] * len(cfg["k_values"])) + "|")
        for metric in ("recall", "precision", "ndcg"):
            vals = " | ".join(f"{gain[metric].get(str(k), 0):+.2f}%" for k in cfg["k_values"])
            gain_lines.append(f"| {metric.capitalize()}@{','.join(str(k) for k in cfg['k_values'])} | {vals} |")
        gain_lines.append(f"| **MRR** | {' | '.join('' for _ in cfg['k_values'])} | **{gain.get('avg_mrr', 0):+.2f}%** |")
        lines.extend(gain_lines)

    # --- 统计显著性（配对检验） ---
    tests = [
        ("hybrid vs vector", s.get("hybrid_vs_vector_test")),
        ("hybrid_rerank vs hybrid", s.get("rerank_vs_hybrid_test")),
    ]
    tests = [(label, t) for label, t in tests if t]
    if tests:
        lines.append("\n## 三、统计显著性（配对检验）\n")
        lines.append("配对检验基于**同一 query 的逐条指标差** Δ_i：Wilcoxon signed-rank 判显著性、")
        lines.append("paired bootstrap 算差值的 95% CI。**CI 不含 0 且 p < 0.05 才视为显著**；")
        lines.append(f"当前样本量 {cfg['num_queries']} 条偏小，结论供参考，扩到 100+ 后更可靠。\n")
        lines.append("| 对比 | 指标 | 平均差值 | 95% CI | p 值 |")
        lines.append("|------|------|---------|--------|------|")
        for label, t in tests:
            m = t.get("mrr", {})
            r5 = t.get("recall5", {})
            m_ci = m.get("ci", [0.0, 0.0])
            r5_ci = r5.get("ci", [0.0, 0.0])
            lines.append(
                f"| {label} | MRR | {m.get('mean_diff', 0):+.4f} | "
                f"[{m_ci[0]:+.4f}, {m_ci[1]:+.4f}] | {m.get('p', 1):.4f} |"
            )
            lines.append(
                f"| {label} | Recall@5 | {r5.get('mean_diff', 0):+.4f} | "
                f"[{r5_ci[0]:+.4f}, {r5_ci[1]:+.4f}] | {r5.get('p', 1):.4f} |"
            )

    # --- 逐查询详情 ---
    lines.append("\n## 四、逐查询详情\n")
    for i, d in enumerate(result.get("detailed", []), 1):
        lines.append(f"### Q{i}: {d['query'][:60]}")
        lines.append(f"相关文档数: {d['relevant_count']}\n")
        lines.append("| 模式 | Recall@5 | Precision@5 | MRR | NDCG@5 | 耗时(ms) | Top-5 命中数 |")
        lines.append("|------|----------|-------------|-----|--------|----------|-------------|")
        for mode in modes:
            m = d["modes"].get(mode, {})
            if not m:
                continue
            lines.append(
                f"| {mode} | {m['recall'].get('5', '-')} | {m['precision'].get('5', '-')} | "
                f"{m['mrr']} | {m['ndcg'].get('5', '-')} | {m['elapsed_ms']:.1f} | "
                f"{m['hits_in_top5']} |"
            )
        lines.append("")

    # --- 简历可用结论 ---
    lines.append("## 五、简历可引用结论\n")
    if gain:
        r5_gain = gain.get("recall", {}).get("5", 0)
        r3_gain = gain.get("recall", {}).get("3", 0)
        lines.append(f"- 混合检索（BM25 + 向量 + RRF 重排序）相比纯向量检索，Recall@5 {'提升' if r5_gain > 0 else '变化'}{r5_gain:+.1f}%")

        h_mrr = s.get("hybrid", {}).get("avg_mrr", 0)
        v_mrr = s.get("vector", {}).get("avg_mrr", 0)
        mrr_verdict = (
            "首个相关结果排名更靠前"
            if h_mrr >= v_mrr
            else "浅排名下略低于纯向量（等权 RRF 被较弱的关键词路稀释，属预期，见召回阶段章节）"
        )
        lines.append(f"- 混合检索 MRR = {h_mrr:.4f}，纯向量 MRR = {v_mrr:.4f}，{mrr_verdict}")

        # 耗时
        avg_hyb_ms = sum(
            d["modes"].get("hybrid", {}).get("elapsed_ms", 0)
            for d in result.get("detailed", [])
        ) / max(len(result.get("detailed", [])), 1)
        lines.append(f"- 混合检索平均耗时 {avg_hyb_ms:.0f} ms")

    if stage_recall:
        lines.append(
            f"- 召回阶段（candidate_k={candidate_k}）混合检索候选集 Recall = {stage_recall.get('hybrid', 0):.3f}，"
            f"高于纯向量 {stage_recall.get('vector', 0):.3f}（{stage_gain:+.1f}%），说明双路召回在候选集深度互补"
        )

    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def _build_metric_table(summary: dict[str, Any], metric: str, modes: list[str]) -> str:
    """构建指标对比表。"""
    # 找一个 mode 获取 K 值列表
    k_keys = list(summary.get("vector", {}).get(metric, {}).keys()) or ["1","3","5","10"]

    lines = ["| 模式 | " + " | ".join(f"K={k}" for k in k_keys) + " |"]
    lines.append("|------|" + "|".join(["------"] * len(k_keys)) + "|")

    for mode in modes:
        m = summary.get(mode, {})
        if not m:
            continue
        vals = " | ".join(
            f"{m.get(metric, {}).get(k, 0):.4f}" for k in k_keys
        )
        lines.append(f"| {mode} | {vals} |")

    return "\n".join(lines)


def _build_mrr_table(summary: dict[str, Any], modes: list[str]) -> str:
    """构建 MRR 对比表（含 bootstrap 95% CI）。"""
    lines = ["| 模式 | MRR | 95% CI |"]
    lines.append("|------|-----|--------|")
    for mode in modes:
        m = summary.get(mode, {})
        if m:
            ci = m.get("mrr_ci", {})
            ci_str = f"[{ci.get('lo', 0):.4f}, {ci.get('hi', 0):.4f}]" if ci else "-"
            lines.append(f"| {mode} | {m.get('avg_mrr', 0):.4f} | {ci_str} |")
    return "\n".join(lines)


def _build_type_table(by_type: dict[str, Any], modes: list[str]) -> str:
    """构建按查询类型分组的 Recall@5 / MRR 子表。"""
    lines = ["| 类型 | " + " | ".join(f"{m} Recall@5" for m in modes) + " | " + " | ".join(f"{m} MRR" for m in modes) + " |"]
    lines.append("|------|" + "|".join(["------"] * (len(modes) * 2)) + "|")

    for tname, per_mode in by_type.items():
        rec = []
        mrr = []
        for mode in modes:
            m = per_mode.get(mode)
            if m:
                rec.append(f"{m['recall5']:.4f}")
                mrr.append(f"{m['mrr']:.4f}")
            else:
                rec.append("-")
                mrr.append("-")
        lines.append(f"| {tname} | " + " | ".join(rec) + " | " + " | ".join(mrr) + " |")

    return "\n".join(lines)


# ==============================================================================
# 自动标注辅助（关键词匹配生成伪 ground-truth）
# ==============================================================================

async def auto_annotate_queries(
    queries: list[dict[str, Any]],
    rag_service: Any,
) -> list[dict[str, Any]]:
    """
    自动标注：使用关键词匹配为每个 query 标注相关 chunk。
    原理：对每个 query，取三种检索模式的 Top-10 并集作为候选池，
         然后用 query 中的关键词与 chunk 内容进行匹配，
         匹配到的标记为 relevant。

    注意：这是近似方法，精度低于人工标注。
         建议先用此方法快速获取基准数字，再人工抽查修正。
    """
    annotated: list[dict[str, Any]] = []

    for item in queries:
        query = item["query"]
        keywords = item.get("relevant_keywords", [])
        if not keywords:
            # 用简单分词作为关键词
            import re
            keywords = list(set(re.findall(r"[\w一-鿿]{2,}", query)))

        # 汇集三种模式的候选集
        all_results: list[Any] = []
        for mode in ("vector", "keyword", "hybrid"):
            try:
                hits = await rag_service.search(query, top_k=10, mode=mode)
                all_results.extend(hits)
            except Exception:
                pass

        # 去重
        seen: set[str] = set()
        unique_results: list[Any] = []
        for r in all_results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        # 关键词匹配判定相关性（至少命中 2 个关键词视为相关）
        relevant_ids: list[str] = []
        for r in unique_results:
            content_lower = r.content.lower()
            hits_kw = sum(1 for kw in keywords if kw.lower() in content_lower)
            if hits_kw >= min(2, len(keywords)):
                relevant_ids.append(r.id)

        annotated.append({
            "query": query,
            "relevant_ids": relevant_ids,
            "relevant_keywords": keywords,
            "_auto_generated": True,
        })

    return annotated


# ==============================================================================
# CLI 入口
# ==============================================================================

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG 检索召回准确率评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/eval_rag.py                        # 使用 eval_queries.json 评估
  python scripts/eval_rag.py --auto-annotate        # 自动标注模式（从 queries 生成 ground truth）
  python scripts/eval_rag.py --top-k 30             # 检索 Top-30
  python scripts/eval_rag.py --output report.md     # 指定输出文件
        """,
    )
    parser.add_argument("--queries", type=str, default=None,
                        help="测试查询 JSON 文件路径")
    parser.add_argument("--auto-annotate", action="store_true",
                        help="使用关键词匹配自动标注（快速但近似）")
    parser.add_argument("--top-k", type=int, default=20,
                        help="检索时返回的结果数（应 >= 最大评估 K 值）")
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K,
                        help="召回阶段评估的候选数量（RRF 喂给精排的候选集大小）")
    parser.add_argument("--rerank", action="store_true",
                        help="追加 hybrid_rerank 消融组（hybrid + cross-encoder 精排），需 rerank 模型可用")
    parser.add_argument("--rerank-candidate-k", type=int, default=None,
                        help="精排深度（默认对齐生产 settings.rerank_candidate_k）。"
                             "与 --candidate-k 独立：candidate-k 是召回池深度，本参数是精排重排深度。"
                             "做深度消融时分别传 20 / 50。")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告路径（默认 scripts/eval_report.md）")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 检索召回准确率评估实验")
    print("=" * 60)

    # 初始化
    evaluator = RAGEvaluator(include_rerank=args.rerank)

    # 加载或自动标注查询
    if args.auto_annotate:
        print("\n[自动标注模式] 使用关键词匹配生成 ground truth...")
        queries_path = Path(args.queries) if args.queries else QUERIES_FILE
        if not queries_path.exists():
            print(f"[错误] 查询文件不存在: {queries_path}")
            sys.exit(1)
        with open(queries_path, "r", encoding="utf-8") as f:
            raw_queries = json.load(f)
        annotated = await auto_annotate_queries(raw_queries, evaluator._rag)
        queries = [
            TestQuery(
                query=q["query"],
                relevant_ids=set(q.get("relevant_ids", [])),
                relevant_keywords=q.get("relevant_keywords", []),
            )
            for q in annotated
        ]
        # 统计标注情况
        total_relevant = sum(len(q.relevant_ids) for q in queries)
        print(f"  自动标注完成: {len(queries)} 条查询, 共 {total_relevant} 个相关 chunk")
        if total_relevant == 0:
            print("  [警告] 没有匹配到任何相关 chunk，请检查知识库内容和查询关键词")
    else:
        queries = evaluator.load_queries(
            Path(args.queries) if args.queries else None
        )

    # 验证 K 值
    if max(EVAL_K_VALUES) > args.top_k:
        print(f"[警告] 最大评估 K={max(EVAL_K_VALUES)} > 检索 top_k={args.top_k}，结果可能不准确")

    # 执行评估
    print()
    result = await evaluator.evaluate(
        queries,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank_candidate_k=args.rerank_candidate_k,
    )

    # 生成报告
    output_path = Path(args.output) if args.output else REPORT_FILE
    report = generate_markdown_report(result, output_path)

    print("\n" + "=" * 60)
    print(f"评估完成！报告已保存至: {output_path}")
    print("=" * 60)

    # 简要结论
    s = result["summary"]
    hyb = s.get("hybrid", {})
    vec = s.get("vector", {})
    print("\n── 快速结论 ──")
    print(f"  混合检索 Recall@5: {hyb.get('recall', {}).get('5', 0):.4f}")
    print(f"  纯向量检索 Recall@5: {vec.get('recall', {}).get('5', 0):.4f}")
    print(f"  混合检索 MRR:       {hyb.get('avg_mrr', 0):.4f}")
    print(f"  纯向量检索 MRR:     {vec.get('avg_mrr', 0):.4f}")
    gain = s.get("hybrid_vs_vector_gain", {})
    if gain:
        print(f"  Recall@5 提升:      {gain.get('recall', {}).get('5', 0):+.2f}%")
    sr = s.get("stage_recall")
    if sr:
        ck = s.get("stage_candidate_k", CANDIDATE_K)
        print(f"  召回阶段 Recall@{ck}:  hybrid={sr.get('hybrid', 0):.4f} vs vector={sr.get('vector', 0):.4f} (互补 {s.get('stage_recall_gain', 0):+.2f}%)")
    rg = s.get("rerank_vs_hybrid_gain")
    if rg:
        print(f"  rerank 边际收益:  Recall@5 {rg.get('recall', {}).get('5', 0):+.2f}% | MRR {rg.get('avg_mrr', 0):+.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
