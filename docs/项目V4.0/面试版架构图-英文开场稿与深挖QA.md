# 心血管慢病管理 AI Agent — 英文开场稿 + 深挖追问 Q&A

> 配套 [面试版架构图.md](面试版架构图.md) 使用。
> 上半部分是英文 60 秒开场稿（外企/海外岗位）；下半部分是把「每层展开点」扩成的深挖追问 Q&A，答案均基于本项目真实设计。

---

## 一、英文 60 秒开场稿

> My project is a **cardiovascular chronic-disease management AI agent**. It tackles three problems: how to wire an LLM into real tasks, how to make retrieval more accurate, and how to keep multi-turn conversations contextual.
>
> First, **task orchestration**. I built a three-tier mode router — rule-based fast path, an LLM router, and a ReAct fallback — that switches between a **ReAct loop** and **Plan-and-Execute** depending on intent and task complexity. Tool invocation runs on Function Calling as a backend, with automatic fallback if it's unsupported, and a **reflection loop** wraps the whole thing: high-risk answers get evaluated, corrected, and retried.
>
> Second, **knowledge augmentation**. I implemented **two-stage retrieval**: BM25 keyword search plus bge-m3 dense retrieval, fused with RRF for recall, then re-ranked by a cross-encoder. In evaluation, **MRR improved 29.9% and Precision@1 by 50%**, at the cost of roughly two seconds of extra latency.
>
> Third, **memory**. Short-term uses a Redis sliding window with LLM summarization; long-term uses Milvus for semantic recall; PostgreSQL persists everything underneath. The three layers together handle cross-session context.
>
> Finally, the system has a **medical safety boundary** — a keyword pre-filter plus LLM confirmation, so only *actively occurring* high-risk symptoms are short-circuited before they reach the tool chain.

**备选短语速查**（防止卡壳）：
- 两阶段检索 two-stage retrieval
- 召回 recall / 精排 rerank (cross-encoder re-ranking)
- 倒数排名融合 Reciprocal Rank Fusion (RRF)
- 滑窗 + 摘要 sliding window + summarization
- 反思闭环 reflection loop
- 安全边界 safety boundary / guardrail

---

## 二、深挖追问 Q&A

### 1. RAG 检索

**Q: RRF 和线性加权（linear fusion）比，好在哪？**
> 线性加权要求两路检索的分数在同一量纲——但 BM25 的分数可能是 0~50，向量余弦相似度是 -1~1 或 0~1，直接加权前必须先做分数归一化，而归一化方案（min-max、z-score）对异常值很敏感。RRF 只看**排名**不看原始分数：`score = Σ 1/(k + rank)`，天然跨量纲可比、无需调归一化；且对单个异常高分鲁棒（一个路线的极端分不会主导结果）；超参 `k`（通常 60）很不敏感，开箱即用。代价是丢掉分数幅度信息——排名第 1 和第 2 哪怕分数差 10 倍，在 RRF 里差别也只是一点点。对本项目「中文医疗文档」这种两路分数分布差异大的场景，RRF 是更稳的默认选择。

**Q: 为什么向量检索还不够，要加 BM25？**
> 纯向量检索对**精确术语、专有名词、编号、药物名**这类词面匹配不敏感——比如用户问「XX 降压药的说明书」，向量可能召回语义相近但没出现该药名的段落。BM25 靠词频/逆文档频率做精确词面召回，和向量语义召回互补：一个抓「字面命中」，一个抓「语义相近」。医疗领域术语密集，双路召回比单路召回命中率明显更高。

**Q: 为什么召回之后还要 cross-encoder 精排？两阶段的价值是什么？**
> 召回用的是 bi-encoder（bge-m3），查询和文档**独立**编码再算相似度，快但只能粗粒度匹配，无法建模 query 和 doc 之间细粒度的 token 交互。精排用 cross-encoder（bge-reranker-v2-m3），把 (query, doc) 拼在一起过一遍 transformer，能学到「这个词和那个词是否真的相关」，精度高得多，但每对 (query, doc) 都要一次前向，代价是 O(n) 次推理，慢。两阶段的本质是**分工**：廉价的 bi-encoder 多召回（候选 50 个），昂贵的 cross-encoder 只在候选集上精排取 top（20 个），兼顾「全」和「准」。

**Q: candidate_k 为什么必须大于 top_k？**
> 精排只能**重排**候选集，不能无中生有。如果 candidate_k == top_k，精排只是在同一批 20 个里换个顺序，召回的「池子」没变大，自然无法提升命中。所以把 candidate_k 设成 50（> top_k=20），让精排有机会从更大的池子里把真正相关的顶上来。这是我们第一版评测 recall 反而下降时踩的坑——candidate_k 和 top_k 相等，精排变成了纯重排。

**Q: 精排的收益和代价怎么权衡？**
> 收益：MRR +29.9%（0.558→0.725）、Precision@1 +50%（0.414→0.621）、Recall@5 +64.6%（0.360→0.593），说明精排既把「第一位」排得更准，又从更大的候选集里把真正相关的文档顶了上来——这对聊天式场景（用户主要看第一条）很有价值。代价是单查询延迟 +2 秒（cross-encoder 逐对打分）。权衡结论是**按部署场景开关**：追求 top-1 准确就开精排；对延迟敏感就关掉或只做候选集上更轻的精排。

### 2. Agent 编排

**Q: 为什么不直接用 LangChain 的 Agent，要自己实现 ReAct？**
> 三个原因。一是**控制力**：LangChain 的 AgentExecutor/ReAct 高度封装，内部 prompt、解析、工具调用流程是黑盒，我想在循环里插入医疗安全门、反思闭环、三级模式路由这些横切逻辑很别扭。二是**稳定性**：LangChain 早期 API 迭代很快、破坏性变更多，自己写一个轻量的 ReAct 循环（Thought→Action→Observation）依赖少、好调试。三是**理解深度**：这是个面试/学习项目，自己实现才能真正讲清楚 ReAct 和 Function Calling 的机制——prompt 怎么组织、工具调用怎么解析、循环怎么终止。说实话，生产环境用框架无可厚非，但这里自研是为了把底层机制吃透、并且精确控制行为。

**Q: ReAct 和 Plan-and-Execute 怎么选？各自什么场景、什么失败模式？**
> **ReAct**：推理和行动交错，每步决定下一步动作。适合短任务、单步工具调用，开销小。失败模式是长任务里「迷失」——没有全局视角，容易重复动作或循环。**Plan-and-Execute**：先让 LLM 产出计划（拆 2~8 个子任务），每个子任务走 ReAct 执行，失败就重规划（≤2 次），最后汇总。适合多步复杂任务，可预测、可解释。开销是规划那一趟的 token/延迟，简单任务不划算。路由规则就是按意图和复杂度选：数学计算/闲聊/知识问答走 react，实时搜索、时间敏感的多步任务走 plan_execute。

**Q: Function Calling 和文本解析（JSON）是什么关系？你说的「后端可降级」是什么意思？**
> Function Calling 是 ReAct 循环里「调用工具」这一动作的**实现后端**，不是第三种模式。FC 让模型直接输出结构化的工具调用（带 JSON schema），解析确定、不会出现正则匹配不到的情况。但有些模型/API 不支持 FC，所以有降级路径：换成文本兼容后端，靠解析 `Action: ...` / `Action Input: ...` 这种约定格式来触发工具。关键点在于**上层模式（react/plan_execute）完全不变**，只是换掉最底层的「工具调用如何落地」——这就是分层的好处。

**Q: 反思闭环怎么防止「越反思越差」或死循环？**
> 三层防护。一是**风险分级（gating）**：不是每个回答都反思，只有高风险（比如医疗建议）才触发，普通闲聊直接跳过，省成本也避免过度纠正。二是**重试上限**：反思修正后最多完整重试 2 次，避免无限循环。三是**触发条件收敛**：反思只在「评估发现问题」时才进入修正路径，评估本身有明确维度（准确性、安全性、完整性），不是每次都重跑。

### 3. 记忆

**Q: 为什么记忆要分三层？不能一个向量库全搞定吗？**
> 因为三类记忆的**访问模式和时效要求不同**。短期记忆要的是「当前这轮对话的最近上下文」，向量语义召回会丢失**精确的先后顺序和最近几轮的原文**，而多轮对话的连贯性恰恰依赖这个，所以用 Redis 滑窗 + token 上限直接存原文，快且有序。长期记忆要的是「跨会话的语义关联」（用户上周问过高血压，这周问饮食，能召回那件事），这才需要向量库做语义检索。PostgreSQL 做持久化冷存储，是可审计、可关系查询的落盘层。三者各司其职，`PersistentMemoryManager` 用 Redis 走热路径、PG 做冷存储、内存做兜底降级。

**Q: 短期记忆的摘要压缩是怎么触发、怎么压的？**
> 滑窗有 token 上限（4000），超出时触发 LLM 摘要。压缩策略是「压缩旧的、保留近的」：把窗口里较早的对话轮次压缩成摘要，**保留尾部最近几轮原文**——因为「recency」对当前回复的连贯性最关键，直接扔掉会断档。这样既缩短了上下文长度、省 token，又不丢最近的关键信息。

### 4. 安全与评测

**Q: 医疗安全边界具体怎么实现的？**
> 两阶段：**关键词粗筛 + LLM 确认**。第一段用领域 policy 的 `risk_keywords`（胸痛、呼吸困难、意识不清等急性症状词）做高召回粗筛——未命中直接放行、零 LLM 成本，命中才进入第二段。第二段用 LLM 把命中项分成三态：`emergency`（正在发作，短路回急诊话术）、`clarify`（时态不明，追问「你现在是否正出现这些症状」）、`safe`（假设/既往/科普，正常回答 + disclaimer）。LLM 不可用或确认失败时，保守降级为 emergency，优先保证「不漏拦正在发作的危险信号」。

**Q: 为什么不用纯关键词，也不用纯 LLM？「如果我胸痛怎么办」会被拦吗？**
> 纯关键词会误伤：子串匹配分不清「我现在胸痛」（该拦）和「如果我胸痛怎么办 / 我以前胸痛 / 胸痛是什么病」（不该拦）。纯 LLM 又慢且不确定，而安全门要低延迟、高可用。所以两段分工：关键词保证「漏拦代价高，宁可多拦」，LLM 只对命中的少数输入做精确确认、把假阳性放行。现在「如果我胸痛怎么办」走 clarify 追问、「我以前胸痛」走 safe 正常回答，只有「我正在胸痛」才短路急诊话术。

**Q: 你怎么评估 RAG 效果？选了哪些指标、为什么？**
> 建了一批带 `relevant_ids` 人工标注的测试查询，跑 A/B（精排开 vs 关）。指标四件套：**Recall@K**（该召回的有没有召回）、**Precision@K**（召回的里有多少是相关的）、**NDCG@K**（排序质量，靠前的相关文档加权）、**MRR**（第一个相关文档的位置）。这四个一起看，因为 Recall 和 Precision 往往此消彼长，只看一个会失真。比如我们精排后 MRR、Precision@1、Recall@5 都涨了——说明精排从更大的候选集里既把「第一位」排准了、又把真正相关的文档召回了更多，这是两阶段「先多召回、后精排」的收益，不是单一指标虚高。

**Q: 这个项目的评测有什么不足，你会怎么改进？**
> 诚实地讲，最初自动标注集比较稀疏、噪声较大——45%（13/29）的查询标注为空、平均每个查询只有 1.55 个相关块，导致基线 Recall@5 只有约 0.2。后来我用 **LLM-as-judge** 重标注：空标注降到 2 个、平均相关块升到 4.83，基线指标随之回升。剩下还能改进的：一是**多人标注 + 交叉校验**，二是加**端到端指标**（不只检索，还要看最终生成答案的质量），三是把延迟纳入决策（延迟 vs 质量的帕累托曲线决定是否开精排）。面试时主动说这些，比只报漂亮数字更可信。

---

## 三、高频追问速记（一分钟扫一遍）

| 追问 | 一句话答案 |
|------|-----------|
| 为什么 RRF 不线性加权 | 两路分数跨量纲，RRF 只看排名、免归一化、鲁棒 |
| 为什么双路召回 | BM25 抓字面命中，向量抓语义，医疗术语密集需互补 |
| 为什么两阶段精排 | bi-encoder 快而粗做召回，cross-encoder 准而慢做精排 |
| candidate_k 为什么 > top_k | 精排只能重排，池子不变则无召回收益 |
| 为什么不直接用 LangChain | 黑盒难插横切逻辑、API 不稳、自研才能讲清机制 |
| ReAct vs Plan-Execute | 短任务 ReAct 省开销，多步 plan 拆解+重规划更稳 |
| FC 可降级指什么 | FC 是工具调用后端，不支持时换文本解析，模式不变 |
| 反思怎么不失控 | 风险分级触发 + 重试上限 + 收敛条件 |
| 记忆为什么三层 | 访问模式不同：原文有序/语义关联/持久落盘 |
| 精排代价 | +2s 延迟换 MRR+29.9%、P@1+50%、Recall@5+64.6%，按场景开关 |
| 医疗安全怎么不误伤 | 关键词粗筛 + LLM 确认，只拦「正在发作」，假设/既往放行 |
