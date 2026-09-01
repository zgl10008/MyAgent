# 评测标注方法论（P1 / P2 版）

本目录是 RAG 评测的全套脚本。标注（ground truth）的质量直接决定评测数字可不可信，
这里把「怎么标、为什么这么标、怎么知道标得好不好」讲清楚。

## 一、文件与脚本速览

| 文件 / 脚本 | 作用 |
|---|---|
| `eval_queries.v2.json` | 现有 29 条查询（含 `type` 标签） |
| `eval_queries.graded.json` | 分级标注输出（`relevance: {id: 1|2}`） |
| `eval_queries_dev.json` / `eval_queries_test.json` | dev（调参）/ test（最终评估）拆分 |
| `eval_queries_expanded.json` | 合并全集 |
| `doc_chunks_full.json` | 全库 916 块（`dump_kb.py` 导出） |
| `eval_metrics.py` | 指标 + 统计（bootstrap / Wilcoxon / Kappa） |
| `eval_rag.py` | 检索层评测主脚本 |
| `eval_generation.py` | 生成层幻觉评测 |
| `reannotate_queries.py` | 二元标注（0/1），旧版 |
| `reannotate_graded.py` | 分级标注（0/1/2） |
| `pooling_bias_check.py` | 全库通读基准 + Kappa 一致性 |
| `expand_queries.py` | 查询扩量 + 拆 dev/test |
| `regression_check.py` | CI 回归门槛（hybrid Recall@5/MRR 红线；`--standalone` = 自包含 smoke 门禁） |
| `calibrate_refuse_threshold.py` | 库外拒答门控阈值校准（精排 top-1 分扫网格） |
| `judge_agreement.py` | LLM-as-judge 一致性校验（分层抽样 + Kappa） |
| `baseline_standalone.json` | CI smoke 门禁的基线（自包含，无 Milvus/bge-m3） |
| `feedback_to_queries.py` | 在线负反馈回流成待标注测试集 |

## 二、分级相关度（0/1/2）——为什么不用二元

二元标注只能算 Recall / Precision / MRR（只看「是否相关」），**体现不了排序质量**。
`NDCG` 需要增益权重：把「高度相关」的块排在第 1 名，应该比「部分相关」排第 1 名加分更多。

分级口径（见 `reannotate_graded.py` 的 `JUDGE_SYSTEM`）：
- `2` = 高度相关：直接回答问题的核心事实 / 数据 / 定义 / 结论
- `1` = 部分相关：提供背景或部分信息，但不完整、不能单独回答
- `0` = 不相关（默认，可省略）

标注产出 `relevance: {chunk_id: grade}`；`relevant_ids = grade > 0` 的集合仍供 Recall /
Precision / MRR 用（这些指标只关心相关与否）。分级只影响 NDCG 的增益权重。

运行：
```bash
aiAgent/Scripts/python.exe scripts/reannotate_graded.py
```

## 三、标注一致性（Cohen's Kappa）——怎么知道标准清不清楚

LLM-as-judge 的标注如果标准模糊，两个独立 judge 会给出分歧很大的结果，评测数字本身就不可信。
Kappa 量化这种一致性（`eval_metrics.cohens_kappa`）：

- `< 0.6`：一致性不足，标注标准模糊，需先对齐提示词 / 换 judge
- `≥ 0.6`：可用
- `≥ 0.8`：良好

用「标准口径」和「严格口径」两个 judge 对同一候选池打 0/1，算逐 query Kappa：
```bash
aiAgent/Scripts/python.exe scripts/pooling_bias_check.py --mode kappa --limit 5
```

## 四、Pooling bias（候选池漏检）——为什么 Recall 会虚高

评测的候选池是「现有检索系统捞出来的」。池外可能还有真正相关的块，标注者永远标不到 → 漏检
被系统性漏掉 → Recall 虚高。解决：用「全库通读」做基准，把 916 块分批复读、LLM 判相关，
对比「池内标注」和「全库标注」的差异。

漏检率 = |全库相关 − 池内相关| / |全库相关|：
```bash
aiAgent/Scripts/python.exe scripts/pooling_bias_check.py --limit 5
```

解读：漏检率 > 20% 的 query 应全库重标；整体偏高则考虑扩候选池（加多路召回 / 提高 Top-K）。

## 五、查询扩量 + dev/test 拆分——为什么不能一套数据又调参又评估

29 条查询类型失衡（18 事实 / 10 多跳 / 1 边界 / 0 对比 / 0 反例），且一套数据既调参又评估会
overfit。`expand_queries.py` 用 LLM 生成缺的类型（对比 / 多跳 / 边界 / 反例），标注后拆：

- **dev**（原 v2）→ 调超参随便碰
- **test**（新增）→ 最终评估，**只碰一次**

```bash
aiAgent/Scripts/python.exe scripts/expand_queries.py --num 12   # 冒烟
aiAgent/Scripts/python.exe scripts/expand_queries.py            # 补齐到 100+
```

「反例 / 库外」查询（如「帕金森病如何治疗」）预期 0 相关——它考的是系统能否诚实说「我不知道」
而不是强行检索，属于生成层的反事实测试，检索层的 Recall 对这类 query 应置 0 处理。

## 六、完整流程（推荐顺序）

```bash
# 1. 导出全库（若知识库变了）
aiAgent/Scripts/python.exe scripts/dump_kb.py

# 2. 分级标注（LLM-as-judge 0/1/2）
aiAgent/Scripts/python.exe scripts/reannotate_graded.py

# 3. 一致性 + pooling bias 检查（抽少量验证标注可信度）
aiAgent/Scripts/python.exe scripts/pooling_bias_check.py --mode kappa --limit 5
aiAgent/Scripts/python.exe scripts/pooling_bias_check.py --limit 5

# 4. 扩量 + 拆 dev/test
aiAgent/Scripts/python.exe scripts/expand_queries.py

# 5. 检索层评测（在 dev 上调参，test 只跑一次）
aiAgent/Scripts/python.exe scripts/eval_rag.py --queries scripts/eval_queries_dev.json
aiAgent/Scripts/python.exe scripts/eval_rag.py --queries scripts/eval_queries_test.json

# 6. 生成层幻觉评测
aiAgent/Scripts/python.exe scripts/eval_generation.py

# 7. CI 回归门槛（P3）：先建基线，之后每次改检索代码一条命令判回退
aiAgent/Scripts/python.exe scripts/regression_check.py --update-baseline   # 建基线（写 baseline.json）
aiAgent/Scripts/python.exe scripts/regression_check.py                     # 判回退（跌破红线 exit 1）

# 8. 在线负反馈回流（P3）：POST /rag/feedback 采集 → 转成待标注测试集
aiAgent/Scripts/python.exe scripts/feedback_to_queries.py                  # data/feedback.jsonl → eval_queries.feedback.json
```

## 七、P4 补全（口径对齐 / 拒答门控 / judge 一致性 / CI smoke）

上一轮评测收尾时发现的 4 个工程缺口，对应的脚本与用法：

### 1. rerank 候选深度口径对齐（--rerank-candidate-k）

生产 `rerank_candidate_k`（默认 20）与评测里的「召回阶段 candidate_k（50）」是两个概念，不能混。
`eval_rag.py` 新增 `--rerank-candidate-k`，把精排深度与召回池深度解耦：

```bash
# 深度消融：精排分别只重排 top-20 / top-50，对比 rerank 边际收益
aiAgent/Scripts/python.exe scripts/eval_rag.py --rerank --rerank-candidate-k 20 --queries scripts/eval_queries_dev.json
aiAgent/Scripts/python.exe scripts/eval_rag.py --rerank --rerank-candidate-k 50 --queries scripts/eval_queries_dev.json
```

消融结论（dev 77 条）：rc20 精排边际 Recall@5 **+11.62%** / MRR **+13.95%**；rc50 **+15.06%** / **+18.98%**。
更深候选池（50）带来更高精排增益，且 916 块库上 cross-encoder 精排 50 候选开销可忽略——故生产
`rerank_candidate_k` 已 **20→50**（`config.py`），评测与生产口径统一。

### 2. 库外拒答门控（RAG_REFUSE_THRESHOLD）

反例（库外）查询被硬答 + 幻觉，faithfulness 掉到 ~0.3。`chat()` 新增「精排 top-1 分低于阈值即拒答」的门控，
阈值在评测集上校准：

```bash
aiAgent/Scripts/python.exe scripts/calibrate_refuse_threshold.py                     # 扫阈值网格，打印推荐值
aiAgent/Scripts/python.exe scripts/calibrate_refuse_threshold.py --verify-refuse 0.4 # 端到端验证反例全拒答
# 把推荐值写入 .env：RAG_REFUSE_THRESHOLD=<推荐值>（留空/注释 = 关闭门控）
```

阈值只对 cross-encoder 精排分（[0,1] 越大越相关）有意义，故仅 rerank 开启时生效。

校准结论（`eval_queries_full` 99 条）：反例 12 条 top-1 分 `max=0.368`，领域内 87 条 `min=0.071`，
**两分布重叠、单阈值无法完全分离**。F1 最优 `0.05`（反例拒答 75% / 领域内误拒 0%）；安全优先 `0.40`
（反例拒答 **100%** / 领域内误拒 ~14%，误拒集中在精排分 <0.40 的低证据查询，硬答本就低忠实度）。
医疗场景建议 `0.40`（宁拒勿幻觉）；`--verify-refuse 0.4` 已确认 12 条反例端到端 100% 拒答（不触发 LLM）。

### 3. LLM-as-judge 一致性校验（judge_agreement.py）

faithfulness 等数字缺一道「judge 是否可信」的锚。两步走（Milvus 单写者锁，串行）：

```bash
aiAgent/Scripts/python.exe scripts/judge_agreement.py --sample             # 分层抽 20 条 → judge_sample.json
# 独立标注 judge_sample.json → 写 judge_gold.json（{"labels":[{"query":"...","faithful":true}]}）
aiAgent/Scripts/python.exe scripts/judge_agreement.py --agreement          # 算 Kappa（≥0.6 视为可信）
```

### 4. CI smoke 回归门禁（regression_check.py --standalone）

真实门禁依赖 Milvus + bge-m3（`models/` 已 gitignore，CI 离线拿不到）。`--standalone` 用自包含
NGramEmbedding + 内存 BM25/RRF，验证「检索算法 + 评测指标 + 回归对比」链路没被改坏，不测真实 bge-m3 质量：

```bash
aiAgent/Scripts/python.exe scripts/regression_check.py --standalone --update-baseline   # 建 baseline_standalone.json
aiAgent/Scripts/python.exe scripts/regression_check.py --standalone                     # 判回退（exit 1 即回归）
```

CI 的 `.github/workflows/ci.yml` 已加一步 `python scripts/regression_check.py --standalone`。


> 注意：所有脚本都走 `.env` 里的 OpenAI 兼容 API（deepseek）。跑之前确认 venv `aiAgent`、
> `.env` 已配置，且 `models/bge-m3`（embedding）与 HF cache 里的 `bge-reranker-v2-m3` 可用。
