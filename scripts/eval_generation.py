# -*- coding: utf-8 -*-
"""
RAG 生成层幻觉评测（离线，LLM-as-judge）

测「找没找到」之后的「找对了但用没用对」——医疗场景红线是忠实度。

四个指标（RAGAS 0.1.x 三项核心指标 + 本项目额外引用准确性）：
  - Faithfulness      : 答案拆 statement，逐个判「是否被检索上下文支撑」→ supported/total
  - Answer Relevancy  : 从答案反向生成 question，算与原始 question 的 embedding 余弦均值
  - Context Relevancy : 相关上下文 chunk 数 / 检索上下文总数（chunk 级 0/1）
  - 引用准确性          : 答案里 [n] 引用所在的句子，是否真被第 n 个 chunk 支撑

数据源：eval_queries.v2.json（含 type 标签）。
生成走生产路径 RAGService.chat()（检索 + 生成 + [n] 引用）。
judge 用 deepseek-v4-flash（推理模型，MAX_TOKENS 要给足思维链额度）。

使用：
  python scripts/eval_generation.py --limit 5    # 先跑 5 条冒烟
  python scripts/eval_generation.py              # 全量
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.core.rag_service import RAGService, _get_embedding_model
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter

QUERIES_FILE = PROJECT_ROOT / "scripts" / "eval_queries.v2.json"
REPORT_FILE = PROJECT_ROOT / "scripts" / "eval_generation_report.md"

MAX_TOKENS = 6000             # 推理模型：思维链 + 最终 JSON 一起计入，要给足
MAX_CONCURRENCY = 4           # 查询级并发（每条内部串行 generate + 4 次 judge）
EMBED_TRUNC = 2000            # 生成 question 的 embedding 时对文本截断（够用即可）

# ---------------------------------------------------------------- judge 提示词
JUDGE_FAITHFUL = (
    "你是医学答案忠实度（Faithfulness）标注员。"
    "把给定答案拆成独立的陈述（statement），逐个判断该陈述是否被「检索上下文」支撑。\n"
    "只输出一个 JSON：{\"statements\": [{\"text\": \"...\", \"supported\": true}]}\n"
    "判定：supported=true 表示该陈述的内容能在上下文里找到依据；"
    "陈述若引入上下文没有的新事实/数字/结论，则 supported=false。"
)

JUDGE_CTX_REL = (
    "你是上下文相关性（Context Relevancy）标注员。"
    "下面是检索到的上下文（每条一个 chunk，全局编号），判断每条上下文是否与回答用户问题相关"
    "（能提供回答问题所需的信息，哪怕只是部分）。\n"
    "只输出一个 JSON：{\"relevant\": [1, 3]}（相关上下文的编号，从 1 开始；没有则 []）"
)

JUDGE_ANS_REL = (
    "你是答案相关性（Answer Relevancy）标注员。"
    "根据给定答案，反向生成 1~3 个「该答案能够回答」的问题（覆盖答案的核心信息点）。\n"
    "只输出一个 JSON：{\"questions\": [\"q1\", \"q2\"]}"
)

JUDGE_CITATION = (
    "你是引用准确性标注员。答案里用 [n] 标注了引用。"
    "逐个判断：引用 [n] 所在的句子，是否真的被「上下文 n」支撑。\n"
    "只输出一个 JSON：{\"verdicts\": [{\"n\": 1, \"supported\": true}]}"
)

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|\n+")
_CITE_RE = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------- 工具函数
def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点 / 换行切句，返回去空白后的非空句。"""
    parts = _SENT_SPLIT.split(text or "")
    return [p.strip() for p in parts if p.strip()]


def _extract_json(text: str) -> dict[str, Any] | None:
    """鲁棒提取 JSON（容忍 markdown 围栏 / 前后缀文本）。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            pass
    return None


def _build_router(settings: Any) -> ModelRouter:
    return ModelRouter([
        ModelConfig(
            model_id=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            priority=0,
            weight=1.0,
        )
    ])


async def _judge(router: ModelRouter, system: str, user: str) -> dict[str, Any]:
    """一次 LLM 判断，返回解析后的 JSON dict；失败重试 3 次，最终 {}。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(3):
        try:
            resp = await router.chat(messages, temperature=0.0, max_tokens=MAX_TOKENS)
            data = _extract_json(resp.content)
            if data is None:
                raise ValueError(f"无法解析 JSON: {resp.content[:120]!r}")
            return data
        except Exception as e:
            print(f"    [judge 重试 {attempt + 1}] {e}", flush=True)
            await asyncio.sleep(1.0)
    return {}


def _fmt_contexts(contexts: list[Any]) -> str:
    lines = []
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}]\n{c.content}")
    return "\n\n".join(lines) if lines else "（无检索上下文）"


def _citing_sentences(answer: str, indices: list[int]) -> dict[int, str]:
    """返回 {引用编号: 包含 [n] 的句子}（可能因切句遗漏部分）。"""
    sents = _split_sentences(answer)
    out: dict[int, str] = {}
    for n in indices:
        marker = f"[{n}]"
        for s in sents:
            if marker in s:
                out[n] = s
                break
    return out


# ---------------------------------------------------------------- 四项指标 judge
async def judge_faithfulness(router: ModelRouter, answer: str, contexts: list[Any]) -> float | None:
    if not answer.strip():
        return None
    user = (
        f"检索上下文：\n{_fmt_contexts(contexts)}\n\n"
        f"待判答案：\n{answer}\n\n请拆陈述并逐个判断。"
    )
    data = await _judge(router, JUDGE_FAITHFUL, user)
    stmts = data.get("statements") or []
    if not stmts:
        return None
    supported = sum(1 for s in stmts if s.get("supported"))
    return round(supported / len(stmts), 4)


async def judge_context_relevancy(
    router: ModelRouter, query: str, contexts: list[Any]
) -> float | None:
    """Context Relevancy = 相关上下文 chunk 数 / 检索到的上下文总数（chunk 级 0/1）。

    原 Ragas 0.1.x 定义为「必需句 / 上下文总句数」，对长上下文会被系统性低估到 ≈0.03
    （5 条上下文 ~50 句里只挑出 1~2 句必需句），数值几乎无区分度。这里改成 chunk 级
    0/1 判定，更稳、可解释（「5 条上下文里几条真正相关」），对应 Ragas 0.2+ 拆分出
    Context Precision/Recall 前更常用的「上下文是否相关」口径。
    """
    if not contexts:
        return None
    user = (
        f"用户问题：{query}\n\n"
        f"检索上下文（共 {len(contexts)} 条）：\n{_fmt_contexts(contexts)}\n\n"
        "判断每条上下文是否相关，输出 JSON。"
    )
    data = await _judge(router, JUDGE_CTX_REL, user)
    relevant = [
        i for i in data.get("relevant", [])
        if isinstance(i, int) and 1 <= i <= len(contexts)
    ]
    return round(len(set(relevant)) / len(contexts), 4)


async def judge_answer_relevancy(
    router: ModelRouter,
    query: str,
    answer: str,
    embed_model: Any,
) -> float | None:
    if not answer.strip():
        return None
    data = await _judge(router, JUDGE_ANS_REL, f"用户问题：{query}\n\n答案：\n{answer}")
    questions = [q for q in data.get("questions", []) if isinstance(q, str) and q.strip()]
    if not questions:
        return None
    try:
        q_vec = np.asarray(
            embed_model.encode(query[:EMBED_TRUNC], normalize_embeddings=True),
            dtype=np.float32,
        )
        sims = []
        for gq in questions:
            g_vec = np.asarray(
                embed_model.encode(gq[:EMBED_TRUNC], normalize_embeddings=True),
                dtype=np.float32,
            )
            sims.append(float(q_vec @ g_vec))
        return round(float(np.mean(sims)), 4)
    except Exception as e:
        print(f"    [embedding 失败] {e}", flush=True)
        return None


async def judge_citations(
    router: ModelRouter,
    answer: str,
    contexts: list[Any],
    cite_indices: list[int],
) -> float | None:
    if not cite_indices:
        return None
    citing = _citing_sentences(answer, cite_indices)
    if not citing:
        return None
    checks = "\n".join(
        f"引用 [{n}]: {citing[n]}" for n in sorted(citing.keys())
    )
    user = (
        f"答案：\n{answer}\n\n"
        f"检索上下文：\n{_fmt_contexts(contexts)}\n\n"
        f"需要核对的引用句子：\n{checks}\n\n"
        "请逐个判断这些引用句子是否被对应上下文支撑。"
    )
    data = await _judge(router, JUDGE_CITATION, user)
    verdicts = data.get("verdicts") or []
    supported = sum(1 for v in verdicts if v.get("supported"))
    if not verdicts:
        return None
    return round(supported / len(verdicts), 4)


# ---------------------------------------------------------------- 主流程
def _mean(vals: list[float | None]) -> tuple[float, int]:
    """非 None 值求均值，返回 (mean, 有效样本数)。"""
    xs = [v for v in vals if v is not None]
    if not xs:
        return 0.0, 0
    return round(float(np.mean(xs)), 4), len(xs)


async def evaluate_one(
    idx: int,
    total: int,
    item: dict[str, Any],
    rag: RAGService,
    router: ModelRouter,
    embed_model: Any,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    query = item["query"]
    qtype = item.get("type", "")

    try:
        resp = await rag.chat(query, chat_history=None, mode=mode, top_k=top_k)
        answer = resp.answer or ""
        contexts = resp.raw_contexts or []
        cite_indices = [c.index for c in (resp.citations or [])]
    except Exception as e:
        print(f"[{idx}/{total}] 生成失败 {query[:30]}: {e}", flush=True)
        return {"query": query, "type": qtype, "error": str(e)}

    faith = await judge_faithfulness(router, answer, contexts)
    ctx_rel = await judge_context_relevancy(router, query, contexts)
    ans_rel = await judge_answer_relevancy(router, query, answer, embed_model)
    cit_acc = await judge_citations(router, answer, contexts, cite_indices)

    print(
        f"[{idx}/{total}] {query[:30]}  ctx={len(contexts)} cite={len(cite_indices)} "
        f"faith={faith} ansrel={ans_rel} ctxrel={ctx_rel} cit={cit_acc}",
        flush=True,
    )
    return {
        "query": query,
        "type": qtype,
        "answer": answer[:400],
        "n_contexts": len(contexts),
        "n_citations": len(cite_indices),
        "faithfulness": faith,
        "answer_relevancy": ans_rel,
        "context_relevancy": ctx_rel,
        "citation_accuracy": cit_acc,
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["queries"] if isinstance(data, dict) and "queries" in data else data
    return list(items)


def render_report(rows: list[dict[str, Any]], mode: str, top_k: int) -> str:
    from collections import defaultdict

    def agg(key: str) -> tuple[float, int]:
        return _mean([r.get(key) for r in rows])

    f_mean, f_n = agg("faithfulness")
    ar_mean, ar_n = agg("answer_relevancy")
    cr_mean, cr_n = agg("context_relevancy")
    ca_mean, ca_n = agg("citation_accuracy")

    # 按类型聚合 faithfulness / citation_accuracy
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("type") or "未分类"].append(r)

    lines: list[str] = []
    lines.append("# RAG 生成层幻觉评测报告\n")
    lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**评测查询数**: {len(rows)}")
    lines.append(f"**生成模式**: {mode} / top_k={top_k}")
    lines.append(
        "**指标**: Faithfulness / Answer Relevancy / Context Relevancy / 引用准确性\n"
    )

    lines.append("## 一、汇总指标\n")
    lines.append("| 指标 | 均值 | 有效样本 | 说明 |")
    lines.append("|------|------|---------|------|")
    lines.append(f"| Faithfulness（忠实度） | {f_mean:.4f} | {f_n} | 答案陈述被上下文支撑的比例 |")
    lines.append(f"| Answer Relevancy | {ar_mean:.4f} | {ar_n} | 反向生成问题与原问题 embedding 余弦均值 |")
    lines.append(f"| Context Relevancy | {cr_mean:.4f} | {cr_n} | 相关上下文 / 检索上下文总数 |")
    lines.append(f"| 引用准确性 | {ca_mean:.4f} | {ca_n} | [n] 引用被对应 chunk 支撑的比例 |")

    lines.append("\n## 二、按查询类型分组（Faithfulness / 引用准确性）\n")
    lines.append("| 类型 | 样本数 | Faithfulness | 引用准确性 |")
    lines.append("|------|-------|-------------|-----------|")
    for tname, rs in by_type.items():
        tf, tn = _mean([r.get("faithfulness") for r in rs])
        tc, tcn = _mean([r.get("citation_accuracy") for r in rs])
        lines.append(
            f"| {tname} | {len(rs)} | {tf:.4f} (n={tn}) | {tc:.4f} (n={tcn}) |"
        )

    lines.append("\n## 三、逐查询详情\n")
    lines.append("| # | 查询 | 类型 | ctx | cite | Faith | AnsRel | CtxRel | CitAcc |")
    lines.append("|---|------|------|-----|------|-------|--------|--------|--------|")
    for i, r in enumerate(rows, 1):
        def f(v: float | None) -> str:
            return "-" if v is None else f"{v:.4f}"
        lines.append(
            f"| {i} | {r.get('query','')[:28]} | {r.get('type','')} | {r.get('n_contexts',0)} "
            f"| {r.get('n_citations',0)} | {f(r.get('faithfulness'))} | {f(r.get('answer_relevancy'))} "
            f"| {f(r.get('context_relevancy'))} | {f(r.get('citation_accuracy'))} |"
        )

    lines.append("\n## 四、答案样例（前 400 字）\n")
    for i, r in enumerate(rows, 1):
        if r.get("error"):
            lines.append(f"### Q{i}: {r.get('query','')}\n\n生成失败：{r['error']}\n")
            continue
        lines.append(f"### Q{i}: {r.get('query','')}")
        lines.append(f"```\n{r.get('answer','')}\n```\n")

    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 生成层幻觉评测")
    parser.add_argument("--queries", type=str, default=None, help="查询 JSON 文件路径")
    parser.add_argument("--mode", type=str, default="hybrid", help="生成所用检索模式")
    parser.add_argument("--top-k", type=int, default=5, help="生成所用检索 top_k")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量，冒烟用）")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY, help="查询级并发")
    parser.add_argument("--output", type=str, default=None, help="输出报告路径")
    args = parser.parse_args()

    settings = get_settings()
    print("=" * 60)
    print("RAG 生成层幻觉评测（LLM-as-judge）")
    print("=" * 60)

    queries_path = Path(args.queries) if args.queries else QUERIES_FILE
    queries = load_queries(queries_path)
    if args.limit > 0:
        queries = queries[: args.limit]
    print(f"查询数: {len(queries)}  |  mode={args.mode}  top_k={args.top_k}")

    rag = RAGService()
    if rag.chunks_count == 0:
        print("[错误] 知识库为空，请先入库。")
        sys.exit(1)
    print(f"知识库块数: {rag.chunks_count}")

    router = _build_router(settings)
    embed_model = _get_embedding_model()

    sem = asyncio.Semaphore(args.concurrency)

    async def run(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await evaluate_one(
                idx, len(queries), item, rag, router, embed_model,
                args.mode, args.top_k,
            )

    rows = []
    for i, item in enumerate(queries, 1):
        rows.append(await run(i, item))

    report = render_report(rows, args.mode, args.top_k)
    output = Path(args.output) if args.output else REPORT_FILE
    output.write_text(report, encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"评估完成！报告已保存至: {output}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
