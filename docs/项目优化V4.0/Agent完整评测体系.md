# Agent 完整评测体系

> 本文把评测从「RAG 检索 + 生成」向上扩展到**整个心血管慢病管理 Agent 的五层**。
> 姊妹篇：[RAG完整评测方案.md](RAG完整评测方案.md) 已覆盖「检索层 + 生成层」并产出实际基线；本文补齐其余层并给出**可复用基建 + 分阶段实施**。
> 核心原则不变：**先分层定位，再量化归因，最后闭环迭代**。

---

## 〇、现状一句话

RAG 检索 + 生成两层已建成分层评测（Recall/NDCG/Faithfulness 等 + bootstrap/Wilcoxon + CI 回归门 + 负反馈回流）。但 Agent 编排、意图识别、记忆、安全、工具五块**基本无评测覆盖**——`AgentOrchestrator`、`AgentService`、`LongTermMemory`、`IntentRecognizer.clarify`、`CalculatorTool` 等是零测试。

本文不重新发明轮子，而是回答一个工程问题：**怎么用 RAG 评测已沉淀的基建，低成本地把评测铺到全项目。**

---

## 一、五层评测总览

```
用户问题
  │
  ├─ ① 意图层   IntentRecognizer ── 分到哪类？置信度？要不要澄清？
  │
  ├─ ② 编排层   ModeRouter → Planner / ReAct → Reflection → Orchestrator
  │             路由对不对？规划全不全？工具调得准不准？反思拦不拦得住幻觉？
  │
  ├─ ③ 检索+生成层（RAG，已建成 ✅）
  │             见 RAG完整评测方案.md：Recall/MRR/NDCG + Faithfulness/引用准确性
  │
  ├─ ④ 记忆层   ShortTerm / LongTerm / PersistentMemory
  │             压缩丢不丢关键事实？长期语义召回准不准？多级存储一致不一致？
  │
  └─ ⑤ 安全层   DomainPolicy / RiskClassifier ── 急诊拦不拦得住？误拦多不多？
       工具层   Calculator / WebSearch / RAGSearch / ToolRouter（横切，可单测）
```

**可归因定位顺序**（端到端差时回退）：
`AgentService 端到端差 → 意图分错没？（①）→ 路由错没？（②）→ 没召回到？（③ RAG Recall）→ 召回了乱编？（③ Faithfulness / ② 反思没拦住）→ 记忆注入错没？（④）→ 该拦的急诊没拦？（⑤）`

---

## 二、可复用评测基建（本方案的核心）

RAG 评测沉淀的组件**大部分与模块无关**，可直接套用：

| 基建 | 在哪 | 套用到哪 |
|------|------|---------|
| `compute_all_metrics`（Recall/Precision/MRR/NDCG） | `scripts/eval_metrics.py` | **任何「返回有序 ID 列表 + gold 相关 ID 集」的模块**——长期记忆召回、ToolRouter 路由，与 RAG `eval_queries.graded.json` 同构 |
| `bootstrap_ci` / `paired_bootstrap_ci` / `wilcoxon_signed_rank` | `scripts/eval_metrics.py` | 任何 A/B 对比：react vs plan_execute、FC 后端 vs 文本后端 |
| `cohens_kappa` | `scripts/eval_metrics.py` | 任何 LLM-as-judge 的可信度验证：反思幻觉、意图、safety |
| `_judge()` + `_extract_json()`（LLM-as-judge 骨架） | `scripts/eval_generation.py` | 复制成 `eval_intent.py` / `eval_reflection.py` / `eval_safety.py` / `eval_planning.py` |
| `RAGEvaluator` 骨架 + `_paired_test` + 报告模板 | `scripts/eval_rag.py` | 改成 `AgentEvaluator`，跑「同一批 query 两个编排模式」配对检验 |
| 分级 judge + Kappa 保障 | `reannotate_graded.py` / `pooling_bias_check.py` | 任何需 LLM 生成 gold 的模块 |
| CI 回归门槛（红线 = CI 下限） | `regression_check.py` + `baseline.json` | 从「RAG Recall@5 红线」扩展成多模块红线（意图准确率、急诊召回率、ReAct 终止率） |
| `FakeLLM` / `FakeRedis` | `tests/conftest.py` | 扩展 `FakeMilvus` / `FakeModelRouter`，离线组装 `AgentService` / `Orchestrator` 集成测试 |

**关键洞察**：长期记忆召回评测 = 会话级向量检索，**可 100% 复用 RAG 检索评测栈**——`eval_queries.graded.json` 的 `relevance:{id:1|2}` 分级结构 → `compute_all_metrics` → bootstrap/Wilcoxon，只把 chunk 换成 memory 条目。

---

## 三、各模块评测设计

> 每模块给出：可量化信号 → ground truth 形态 → 指标。judge 类型分两种：**确定性比对**（gold 是人工标签/表达式结果）和 **LLM-as-judge**（gold 需模型判断）。

### 3.1 意图层（`app/core/intent/recognizer.py`）

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `intent`（5 类）/ `confidence` / `slots` / `clarify` | `eval_intents.json`：`{query, history?, gold_intent, gold_slots?, should_clarify?}` | 分类准确率 / macro-F1 / 混淆矩阵（定位误分类）；置信度校准 ECE；clarify 触发 P/R |

- **多轮指代消解**：追问样本（"那它呢？"）gold 意图，对比有无 history 的差异。
- **覆盖现状**：`test_intent_recognizer.py` 有关键词/LLM 分类/降级/history 注入，**缺 clarify + 阈值行为**。

### 3.2 编排层

**ModeRouter（`mode_router.py`）**

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `mode` / `source`（rule / llm_router / fallback） | gold mode per query（5 类意图 × 复杂度） | 模式路由准确率；LLM Router 触发率（规则快路径覆盖率）；时间敏感升级 / 非法 mode 降级 / 失败兜底三分支 |

**Planner（`planner.py`）**

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `SubTask` 分解 / `expected_tools` / `replan` / `final_answer` | gold 子任务分解（人工标"应拆成几步"） | **Plan Coverage / Faithfulness**（LLM-as-judge：是否覆盖必要子任务、是否引入无关步骤）；replan 有效性（注入失败→断言 replan 排除失败点）；汇总忠实度（final_answer 只基于 all_results） |

**ReActAgent（`react_agent.py`）**

| 信号 | ground truth | 指标 |
|------|-------------|------|
| action 序列 / 解析错误 / 是否给出 Final Answer / 步数 | gold 工具集 per query | 工具调用 Precision/Recall；终止成功率；平均步数（效率）；解析错误率 |

- 确定性可离线测（FakeLLM 剧本驱动），也是 FC 后端「~0% 解析失败」卖点（`react_agent.py` docstring 自称文本后端 ~10-15% 失败）的可验证点。

**Reflection（`reflection.py`）**

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `action_taken`（none / self_correction / retry_needed）/ `final_score` / `warnings` | gold 幻觉标注（哪些 answer 含虚构事实） | 幻觉检测 Recall/Precision；纠错收益 Δscore（修正前后）；完整性判定准确率 |

- **LLM-as-judge 复用度最高**：幻觉检测直接用 RAG 的 hallucination 标注范式 + `cohens_kappa` 验一致性。

**Orchestrator（`orchestrator.py`）** — 零测试，编排主线

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `mode_used` / `degraded` / `success` / `steps` 里 `backend_fallback` 事件 / `correction_applied` | gold mode + 注入故障 | 模式选择准确率；降级正确率（注入 FC 失败→断言 degraded=True 且仍产出 answer）；反思闭环是否生效；trace 事件完整性（`InMemoryTracer`） |

**AgentService（`agent_service.py`）** — 零测试，端到端核心缺口

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `AgentChatResponse` 全字段 / 安全短路 / 降级兜底 / disclaimer 追加 | FakeLLM+FakeMilvus+FakeModelRouter 组装 | 端到端字段一致性（steps→StepInfo 转换、失败兜底、安全短路不进编排）——把上面所有模块串起来的**集成评测点** |

### 3.3 记忆层

**ShortTermMemory（`short_term.py`）**

| 信号 | ground truth | 指标 |
|------|-------------|------|
| 窗口/压缩/keep 数/摘要 | gold 关键事实集 | 压缩边界（确定性，已覆盖）；**压缩保真度**（LLM-as-judge：摘要保留关键事实、无新增虚构）；注入召回（get_relevant 覆盖最近 N 轮关键上下文） |

**LongTermMemory（`long_term.py`）** — 零直接测试

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `recall` 返回语义相关 memory | `eval_memories.graded.json`：`{query → relevant_memory_ids}` | **Recall@K / Precision / MRR**（直接复用 RAG 栈）；`forget` 确定性单测 |

**PersistentMemory + MemoryManager（`persistent_memory.py` / `manager.py`）**

| 信号 | ground truth | 指标 |
|------|-------------|------|
| LTM 插最前 / limit 截断 / 降级链路顺序 / 多级一致性 | 确定性 | 注入准确性；多级一致性（PG/Redis/内存三份一致）；降级顺序（Redis 失败→PG→内存） |

### 3.4 安全层（`domain/policy.py` + `risk_classifier.py`）

医疗安全是**召回优先**场景，直接借 RAG Recall 思路：

| 信号 | ground truth | 指标 |
|------|-------------|------|
| `disposition`（emergency / clarify / safe） | gold disposition 标注 | **急诊拦截 Recall（漏拦率，必须全拦）+ Precision（误拦率，假设句/既往史/概念咨询不误拦）**；三分类混淆矩阵；`cohens_kappa` 验 judge |

- 额外：**假阴性探测**——用 paraphrase 高危症状（"胸口像压了块石头"不含字面关键词）测 `is_high_risk` 是否漏拦（这是接 LLM 二次确认的动机）。

### 3.5 工具层（可单测为主）

| 模块 | 指标 |
|------|------|
| CalculatorTool（零测试） | 正确性 + 安全性：gold 表达式→结果；边界（除零、`__import__`、函数调用应被拒） |
| WebSearchTool | HTML fixture 集（验证码页/空页/噪声 JSON 页）→ gold 解析结果 |
| ToolRouter（零测试，注意当前 AgentService 未启用它） | 路由准确率：gold 工具集 per query → Precision/Recall（确定性） |

---

## 四、ground truth / judge 一览

| 模块 | ground truth 形态 | judge 类型 |
|------|------------------|-----------|
| 意图分类 | gold 5 类标签 + slots + should_clarify | 确定性比对 |
| 模式路由 | gold mode per query | 确定性比对 |
| ReAct 工具决策 | gold 工具集 per query | 确定性比对 |
| Plan 质量 | gold 子任务分解 | LLM-as-judge（Coverage/Faithfulness） |
| 反思幻觉 | gold 幻觉标注 | LLM-as-judge + Kappa |
| 短期压缩 | gold 关键事实集 | LLM-as-judge + 确定性边界 |
| 长期召回 | gold `{query→memory_ids}`（同构 RAG） | `compute_all_metrics` |
| 医疗安全 | gold disposition | 确定性比对（召回优先） |
| 工具 | gold 表达式/解析/工具集 | 确定性单测为主 |

---

## 五、分阶段实施（按「风险 × 空白」排序）

| 阶段 | 动作 | 产出 |
|------|------|------|
| **P0（完全空白、风险最高）** | `AgentOrchestrator` / `AgentService` 集成测试；`IntentRecognizer.clarify`；`LongTermMemory` 召回评测（复用 RAG 栈）；`CalculatorTool` 安全测试 | 编排主线 + 记忆召回 + 意图 clarify 覆盖 |
| **P1（有核心逻辑但分支缺）** | `PlannerAgent.run_with_replan`；`ModeRouter` 时间敏感/非法/兜底；`ReActAgent` max_steps/非法工具/异常；`ToolRouter` | 编排分支全覆盖 |
| **P2（锦上添花）** | `LangGraphReActAgent`；`MemoryManager`；`ModelRouter`；LLM-as-judge 各模块（意图/反思/planning/safety 的离线 eval） | 全项目闭环 |

**落地路径**（复用基建的最小改动）：
1. 先抽 `scripts/eval_intent.py`（确定性，gold 标签 → F1/混淆矩阵）——意图层最快见效。
2. 再抽 `scripts/eval_safety.py`（gold disposition → 急诊召回优先）——安全是医疗红线。
3. 长期记忆直接复用 `eval_rag.py`：写 `scripts/eval_memory.py` 读 `eval_memories.graded.json`。
4. 编排/反思用 `eval_generation.py` 的 `_judge()` 骨架写 `eval_planning.py` / `eval_reflection.py`。
5. 把 `regression_check.py` 从「RAG 红线」扩成「多模块红线」。

---

## 六、简历可引用结论

- 把「只测 RAG」升级为**覆盖 Agent 五层（意图/编排/检索+生成/记忆/安全）的分层评测体系**，端到端差能按层回退定位。
- 核心工程判断：**长期记忆召回评测 100% 复用 RAG 检索评测栈**（同构的 `{query→relevant_ids}` 分级标注 → `compute_all_metrics` → bootstrap/Wilcoxon），把评测从检索层推广到全项目几乎零新基建。
- 医疗安全的评测是**召回优先**（急诊漏拦率 > 误拦率），与检索的 Recall@K 同思路。
- 用 FakeLLM/FakeMilvus 把 `AgentOrchestrator`/`AgentService` 从「零测试」补成可离线复现的集成评测。
- CI 回归门槛从「RAG 召回红线」扩展成「意图准确率 / 急诊拦截召回率 / ReAct 终止率」的多模块红线。
