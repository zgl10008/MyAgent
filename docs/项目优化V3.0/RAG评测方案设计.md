# RAG 评测方案设计

> 用途：把当前"只有检索层指标"的评测，升级为**分层、可归因、可回归**的完整评测体系。
> 原则一句话：**先分层定位，再量化归因，最后闭环迭代——不能只看一个端到端分数就下结论。**

---

## 〇、背景与目标

当前项目有一个能跑的检索评测（`scripts/eval_rag.py`），但它只回答了"检索找没找到"这一个问题，且有几个结构性缺口。本方案的目标是：

1. **分层**：检索层 / 生成层 / 端到端 三层分开测，端到端差时能回退定位到具体环节。
2. **可归因**：消融实验 + 统计显著性，能说清"每改一步，指标涨在哪、值不值"。
3. **可回归**：评测可重复、可进 CI，改 chunker / reranker / embedding 后一条命令知道"召回有没有回退"。

---

## 一、现状盘点（gap 分析）

| 能力 | 现状 | 结论 |
|------|------|------|
| 检索层指标 | ✅ 已有 Recall@K / Precision@K / MRR / NDCG@K，K=[1,3,5,10] | 保留 |
| 三模式对比 | ✅ vector / keyword / hybrid | 保留 |
| 召回阶段覆盖度 | ✅ stage_recall@candidate_k=50 | 保留 |
| 自动标注 | ✅ 关键词 ≥2 命中（`auto_annotate_queries`） | 保留，作冷启动 |
| **生成层指标（幻觉率）** | ❌ 无 faithfulness / answer relevance | **新增** |
| **rerank 边际收益评测** | ❌ `rerank_enabled=False`，评测里没有 rerank 组 | **新增消融组** |
| **引用准确性** | ❌ generator 强制 `[n]` 引用，但没测引用对不对 | **新增** |
| **统计显著性** | ❌ 只报平均值，无置信区间/显著性 | **新增** |
| **dev/test 分离** | ❌ 29 条 query 一套数据既调参又评估 | **新增** |
| **bad case 归因** | ❌ 报告只有数字，没有"漏检列表" | **新增** |

---

## 二、评测分层框架

三层各测各的，且**每层都要能回退到上一层定位**：

```
用户问题
  │
  ├─ ① 检索层：召回的相关文档全不全、准不准、排序好不好  （离线，ground truth）
  │    指标：Recall@K / Precision@K / MRR / NDCG@K / stage_recall
  │
  ├─ ② 生成层：基于给到的上下文，答案忠不忠实、有没有幻觉  （离线，LLM-as-judge + 人工抽检）
  │    指标：Faithfulness / Answer Relevance / Context Relevance / 引用准确性
  │
  └─ ③ 端到端：用户视角最终答案质量                      （离线人工 + 在线隐式反馈）
       指标：正确性 / 完整性 / 可读性 / 用户是否追问·否定·点击引用
```

**定位顺序**：端到端差 → 先看检索层 Recall（是不是没召回）→ 再看生成层 Faithfulness（是不是召回了但乱编）。

---

## 三、测试集构建（全链路最难、最易被低估）

### 3.1 Query 集：覆盖问题类型，不能只有 happy path

心血管领域按类型分层采样（`scripts/eval_queries.v2.json` 已有 29 条，需扩充并**打类型标签**）：

| 类型 | 示例 | 测什么 |
|------|------|--------|
| 事实型 | "高血压诊断标准是多少" | 单点知识 |
| 对比型 | "高血压 vs 低血压用药差异" | 跨 chunk 召回 |
| 多跳/推理 | "冠心病合并糖尿病首选什么降压药" | 拼接多个 chunk |
| 边界/歧义 | "胸口疼是怎么回事" | 排序质量（一词多病） |
| 反例 | 库外知识、否定句、口语化 | 抗噪/诚实回答 |

> 每条 query 加一个 `type` 字段，报告按类型出子表——这样能看出"哪个类型掉分"，而不是笼统一个平均分。

### 3.2 Ground truth：pool-based + 分级相关

- **候选池**：三种模式（vector/keyword/hybrid）Top-10 并集去重 → 组成候选池。
- **标注对象**：标注者只在池内判断相关度。
- **分级相关**（不要二元）：`0 不相关 / 1 部分相关 / 2 高度相关`。
  - 二元相关只能算 Recall/Precision/MRR；分级相关才能算 NDCG 测排序质量。

### 3.3 必须正视的两个坑

**坑① pooling bias（池外漏检）**：候选池是"现有检索系统捞出来的"，池外可能还有真正相关的文档没进池，标注者永远标不到 → 漏检被系统性漏掉 → **Recall 虚高**。

- 缓解：① 池子用**多路并集**（已做）；② 抽 5~10 条 query 让标注者**全库通读标注**做基准，量化 pool 标注误差；③ 上线后用真实 query 回流补充。

**坑② 标注一致性**：两人标同一批会分歧。算 **Cohen's Kappa**，< 0.6 说明标准不清晰，先对齐标准再标，而不是硬标。

### 3.4 自动标注的定位

`auto_annotate_queries`（关键词 ≥2 命中）是**近似**方法，精度低于人工。定位为：
- **冷启动**：快速拿基准数字。
- **候选池预筛**：给人工标注缩小范围。
- **不能做最终评测**：用有偏的规则去评有偏的系统，误差会叠加。

---

## 四、指标体系

### 4.1 检索层（离线，已实现，保留 + 扩展）

| 指标 | 公式 | 回答的问题 |
|------|------|-----------|
| Recall@K | `|topK ∩ rel| / |rel|` | 找没找全 |
| Precision@K | `|topK ∩ rel| / K` | 找得准不准 |
| MRR | `1 / rank(first rel)` | 首个正确答案早不早 |
| NDCG@K | `DCG/IDCG`，位置权重 `1/log2(rank+1)` | 整体排序质量 |
| stage_recall@candidate_k | 精排前候选集命中相关数 ÷ 相关总数 | 召回环节是否漏货（定义见下） |

四个指标**必须一起报**：Recall 和 Precision 矛盾、MRR 只看第一个、NDCG 看整体，单报一个会掩盖问题。

**stage_recall 精确定义**：本系统是两阶段检索——第一阶段"召回"（BM25 + 向量 → RRF 融合）先把候选集扩到 `candidate_k`（当前 =50）个，第二阶段 cross-encoder 精排到 `top_k`。`stage_recall@candidate_k` = 这 `candidate_k` 个候选里命中的相关 chunk 数 ÷ 该 query 的相关 chunk 总数。

- 它与 `Recall@K` 的区别：`Recall@K` 测**精排后**的浅排名；`stage_recall` 测**精排前**的候选集覆盖度。
- 为什么要单独测：混合检索的增益体现在"候选集覆盖度"这个深度（双路互补能把更多相关 chunk 捞进候选池），而不是最终 top-5——浅排名下等权 RRF 会被较弱的关键词路稀释，所以只看 Recall@5 会低估混合检索的价值。

### 4.2 生成层（离线，**需新增** `scripts/eval_generation.py`）

这一层不测"找没找到"，测"找到的用没用对"，核心是**忠实度**（医疗场景红线）。

**RAGAS 指标（锁定版本：以下定义对应 Ragas 0.1.x 的三项核心指标，实现时 pin 库版本）**：

| 指标 | 精确定义 | 实现方式 |
|------|---------|---------|
| **Faithfulness（忠实度）** | 把答案拆成若干 statement，统计"能被检索上下文支撑"的比例：`faithfulness = \|supported\| / \|total statements\|` | LLM 拆 statement → 逐个判 supported/unsupported |
| **Answer Relevancy（答案相关性）** | 用 LLM 从答案**反向生成**若干 question，再算这些 question 与原始 question 的 embedding 余弦相似度均值 | LLM 生成 question → `mean(cos_sim(gen_q_i, orig_q))` |
| **Context Relevancy（上下文相关性）** | 用 LLM 从上下文里挑出"回答该 question 所必需"的句子，`= \|必需句\| / \|上下文总句数\|` | LLM 挑必需句 → 算句子占比 |

> 版本说明（避免混淆）：
> - 以下三项（Faithfulness / Answer Relevancy / Context Relevancy）是 **Ragas 0.1.x** 的核心指标。
> - **当前 Ragas（0.2+）已大幅演进**：`Context Relevancy` 基本被拆成 `Context Precision` + `Context Recall`，并新增 `Context Entity Recall`、`Noise Sensitivity`、`Answer Correctness` 等，命名与算法也有微调。**不要把"早期三指标"和"当前 Ragas 指标"当成一回事。**
> - 工程落地时**锁定具体 Ragas 版本**（`requirements` 里 pin），指标名以该版本文档为准，评测脚本里记录 `ragas.__version__`。
> - 算法细节：Answer Relevancy 是"反向生成问题 + 算 embedding 余弦相似度均值"，**不是**"让 LLM 直接打分切不切题"；Context Relevancy（0.1.x 定义）是"句子级必需句占比"，**不是**"判断整段是否相关"。

**本项目额外加一项（不属于 RAGAS）**：

| 指标 | 定义 | 实现 |
|------|------|------|
| **引用准确性** | generator 强制 `[n]` 引用，`[n]` 是否真的指向支撑该结论的 chunk | 解析答案里的 `[n]` → 把第 n 个 chunk 回喂 judge → 判是否支撑该句 |

- **人工抽检校准**：LLM-as-judge 本身有偏，抽 20% 样本人工复核，算 judge 与人工的一致性，不一致就调 judge prompt。

### 4.3 端到端（离线人工 + 在线隐式反馈）

- 离线：人工/LLM 综合打分（正确性、完整性、可读性 1~5 分）。
- 在线：隐式负反馈——用户**连续追问同一问题、否定回答、明确说"不对"**，就是离线没测到的真实失败。这些 query 回流到测试集。

---

## 五、实验设计：消融 + 基线

**控制变量（全程固定，否则增益无法归因）**：embedding 模型（bge-m3）、chunk 策略（512/64）、RRF 的 `k=60`、精排的 candidate_k、评测 top-k、ground truth 版本。

**每组唯一变动**（相邻组对照，每组只新增一个变量）：

| 组 | 向量路 | BM25路 | 融合方式 | 重排 | 本次新增 | 对照 |
|----|--------|--------|----------|------|----------|------|
| A | ✅ | — | — | — | 基线（纯向量） | — |
| B | — | ✅ | — | — | 通路换成 BM25 | A vs B：单路差异 |
| C | ✅ | ✅ | 分数归一化 + 线性加权 | — | **新增双路召回**（简单融合基线） | A/B vs C：双路整体增益 |
| C′ | ✅ | ✅ | RRF（k=60） | — | **融合方式换成 RRF** | C vs C′：RRF 独立贡献 |
| D | ✅ | ✅ | RRF（k=60） | ✅ | **新增重排** | C′ vs D：重排边际收益 |

> **归因纪律**：
> - **C 组测的是"双路召回"的整体增益，不是 RRF 的独立贡献**——C 相比 A/B 同时引入了"第二路召回"和"简单融合"两个因素。
> - **只有 C vs C′（融合方式唯一不同）才能说清 RRF 单独值不值**。C′ 是"分数归一化 + 线性加权"的融合基线（向量余弦分 [0,1] 与 BM25 分无上界、量纲不同，需先归一化再线性组合），用来和 RRF 公平对照。
> - 只比相邻组，差异才可归因到"新增的那一个组件"；跨组比（如 A vs D）会混入多个变量。

> D 组目前测不了，因为 `config.py` 里 `rerank_enabled=False`。要先把 rerank 默认打开（模型 `bge-reranker-v2-m3` 已缓存于 HF cache、离线可加载），评测脚本加一个 `mode="hybrid_rerank"` 分支，才能回答"重排值不值"。

---

## 六、统计严谨性

1. **置信区间（默认首选）**：bootstrap 有放回重采样（≥1000 次）算指标 95% CI，报告写"MRR = 0.7251 [0.695, 0.754]"，而不是裸数字。这是 IR 检索评测的标准默认（`ir_measures` / `trec_eval` 都用它）。**模式间差异的 CI 用 paired bootstrap——对"逐 query 的指标差 Δ_i"重采样，不是把两组结果当独立样本混合重采样。**
2. **显著性检验（默认首选）**：**同一批 query、逐 query 配对**。每个 query 分别在两个模式下各跑一次，得到一对指标，检验对象是"逐 query 的指标差 Δ_i = metric_mode2(i) − metric_mode1(i)"（同 query 才可配对，禁止跨 query 或跨模式池化）。用 **Wilcoxon signed-rank test**（非参数，不假设正态）。
   - **为什么不用 paired t-test**：检索指标（Recall/MRR/NDCG）有界在 [0,1]、常呈偏态、方差随均值变化，违背 t 检验的正态性假设，把它当默认会误判显著。
   - 替代：randomization / permutation test（无分布假设），作为需要更严格结论时的升级。
3. **样本量诚实**：29 条能看趋势，但下"显著提升"结论样本偏小。短期靠 bootstrap CI 表达不确定性，中期扩到 100+ 条再做严格显著性。
4. **dev/test 分离**：测试集不能反复拿来调参（会 overfit）。当前 29 条太少，硬分不稳，**先用全量 bootstrap**，扩到 100+ 后拆 `dev.json`（调超参）+ `test.json`（最终评估，只碰一次）。
5. **可复现**：固定随机种子、固定 embedding 版本、固定 ground truth 版本号（现有 `eval_queries.v2.json` 带 `_reannotated: true` 就是版本化意识，继续沿用）。

---

## 七、落地工程化

### 7.1 文件结构（新增/调整）

```
scripts/
  eval_rag.py               # 检索层（现有，扩展：加 rerank 组 + bootstrap + bad case 列表）
  eval_generation.py        # 新增：生成层（faithfulness / relevance / 引用准确性）
  eval_metrics.py           # 新增：抽取共享指标函数（compute_all_metrics、bootstrap、ttest）
  eval_queries_dev.json     # 新增：调参集（扩量后）
  eval_queries_test.json    # 新增：最终评估集（扩量后）
  eval_queries.v2.json      # 现有：29 条，继续扩充并打 type 标签
  HOW_TO_ANNOTATE.md        # 现有：补分级标注 + Kappa 一致性说明
```

### 7.2 回归门槛（CI 可接入）

每次改检索相关代码，一条命令出结论：

```
python scripts/eval_rag.py          # 检索层回归
python scripts/eval_generation.py   # 生成层回归
```

**回归红线**：`hybrid` 的 Recall@5、MRR 不得低于上次基线（在 bootstrap CI 内允许浮动，跌破 CI 下限即报警）。

### 7.3 bad case 归因（报告新增章节）

报告末尾追加"漏检 query 列表"：每条 query 列出"相关但没召回的 chunk + 它被哪个环节丢了（召回还是重排）"，人工逐条分析根因（分词问题 / embedding 覆盖 / 重排误杀），而不是只看平均分。

---

## 八、实施步骤（分阶段，按投入产出排序）

| 阶段 | 动作 | 涉及文件 | 产出 |
|------|------|---------|------|
| **P0** | 打开 rerank 默认 + 评测加 `hybrid_rerank` 消融组 | `config.py`、`eval_rag.py` | 回答"重排值不值" |
| **P0** | 抽取 `eval_metrics.py` + 加 bootstrap CI + Wilcoxon signed-rank | `eval_rag.py` | 数字有误差棒、能判显著性 |
| **P1** | 新增 `eval_generation.py`（faithfulness / relevance / 引用准确性） | 新增文件 | 生成层幻觉评测 |
| **P1** | query 集打 `type` 标签 + 报告按类型出子表 | `eval_queries.v2.json`、`eval_rag.py` | 定位"哪个类型掉分" |
| **P2** | query 扩量到 100+，拆 dev/test | 新增 json | 防止 overfit |
| **P2** | 分级标注 + Kappa 一致性 + 全库通读基准 | `HOW_TO_ANNOTATE.md`、标注数据 | 量化 pooling bias |
| **P3** | 接入 CI 回归门槛 + 在线隐式负反馈回流 | CI、日志 | 闭环迭代 |

---

## 九、一句话总结

> 评测分三层（检索 / 生成 / 端到端），检索层已建、生成层缺失；测试集要分级标注并正视 pooling bias；指标四件套交叉看 + stage_recall 定位召回深度；实验靠消融（尤其补上 rerank 组）归因；结论靠 bootstrap CI + Wilcoxon signed-rank 保证显著；最终靠 bad case 回流和 CI 回归形成闭环。**先补 rerank 评测和统计显著性（P0），再建生成层幻觉评测（P1）。**
