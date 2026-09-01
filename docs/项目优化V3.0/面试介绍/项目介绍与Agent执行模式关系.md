# 项目介绍：通用 Agent/RAG 平台（面试技术参考）

> 本文是面试用的技术参考文档，配合 [README.md](./README.md)（面试讲解稿）使用。
> README 侧重"怎么讲"，本文侧重"怎么理解"——把项目架构和概念关系理清楚。
>
> 项目来源（不主动提，被问时如实说）：基于 GitHub 开源项目 [ai-agent-interview-guide](https://github.com/bcefghj/ai-agent-interview-guide) 做二次开发。

---

## 一、项目定位

这是一个**通用 Agent/RAG 平台**，当前以"**心血管慢病管理**"作为默认 Demo，同时支持上传任意领域的 PDF、Markdown 和文本文件，自动构建对应知识库并进行问答。

项目的目标不是简单封装一次大模型 API，而是构建一个具备以下能力的完整 Agent 系统：

- 工具调用；
- 知识检索；
- 多步任务编排；
- 短期和长期记忆；
- 故障降级；
- 领域安全控制；
- Agent 执行过程追踪。

整体定位可以概括为：

```text
通用 Agent/RAG 底座
+
心血管慢病管理默认 Demo
+
可插拔领域安全策略
```

---

## 二、项目整体能力

项目主要提供三种对话模式。

### 1. Direct 模式

直接调用大模型，适合普通闲聊或不需要外部信息的问题。

### 2. RAG 模式

用户上传文档后，系统进行文档解析、分块、向量化和存储。提问时通过混合检索找到相关内容，再生成带引用的回答。

RAG 在系统中既是一个**独立的问答能力**，也是 Agent 的一个**可调用工具**（`rag_search`）。

### 3. Agent 模式

Agent 根据用户问题进行意图识别和模式路由，然后自主选择工具。目前主要包含：

- RAG 文档检索；
- Web Search；
- Calculator；
- 多模型路由；
- ReAct；
- Plan-and-Execute；
- Function Calling。

---

## 三、一次 Agent 请求的处理流程

```text
用户请求
  ↓
请求校验与领域策略识别（DomainPolicy：高风险关键词 → 安全分流）
  ↓
意图识别（LLM 分类 + 关键词降级双通道）
  ↓
Agent 模式路由（ModeRouter：规则快路径 / LLM Router / react 兜底）
  ├── ReAct
  │    └── _invoke_react()
  │         ├── FunctionCallAgent（LangChain bind_tools，优先路径）
  │         └── ReActAgent（正则解析版，降级后备）
  │              └── 调用 RAG / Web Search / Calculator 工具
  │
  └── Plan-and-Execute
       ├── Planner：拆解任务
       ├── Executor：每个子任务复用 _invoke_react()
       │    └── FunctionCallAgent 或 ReActAgent
       ├── Replanner：失败后重新规划
       └── Synthesizer：汇总结果
  ↓
反思闭环（Evidence 收集 → 评估 → 三级响应）
  ├── ≥80 分 → 直接返回
  ├── 60-79 分 → SelfCorrector 单次 LLM 修复
  └── <60 分 → 触发 Agent 重试（携带反思反馈）
  ↓
四层记忆写入（L1 内存 | L2 Redis | L3 PG | L4 Milvus，并行旁路）
  ↓
返回最终答案 + 执行轨迹（trace_id / steps / tool_calls / reflection_score）
```

---

## 四、RAG 实现

### 1. 文档处理流程

```text
文件上传 → 格式与安全校验 → 文本解析 → 文档分块 → Embedding → 写入向量库和数据库
```

支持 PDF、TXT、MD 和 Markdown 文件。

### 2. 混合检索

- 向量语义检索（Milvus Lite + all-MiniLM-L6-v2, 384-dim COSINE）
- BM25 关键词检索（Okapi + jieba 中文分词）
- RRF 倒数排名融合（k=60）
- 可选的重排序能力

### 3. 向量存储抽象

定义了 `VectorStore` 抽象协议，当前使用 Milvus Lite。RAG 业务层不直接依赖具体 Milvus 类型，未来可替换为 pgvector、Milvus Server 等。

### 4. 失败降级链

```text
sentence-transformers → 失败 → 字符 n-gram Embedding
Milvus Lite → 失败 → 内存向量存储（暴力搜索）
```

---

## 五、Agent 实现

### 1. 意图识别

五类意图：知识问答 / 实时搜索 / 数学计算 / 文档管理 / 闲聊对话。

双通道：LLM 分类（注入对话历史处理指代消解）+ 关键词规则降级（LLM 不可用时兜底）。

### 2. 模式路由

- 规则快路径：高置信度 + 明确映射 → 不调 LLM
- LLM Router：低置信度 / 模糊场景 → 二次判断
- react 兜底

### 3. 工具调用

工具通过统一注册机制接入 Agent。两个后端实现：

| | FunctionCallAgent | ReActAgent |
|---|---|---|
| 状态 | 优先路径 | 降级后备 |
| 工具调用 | LangChain `bind_tools()` → 结构化 JSON | 正则解析 `Thought:` / `Action:` 文本 |
| 优势 | 解析失败率 ~0% | 不依赖模型 function calling 能力 |
| 劣势 | 需要模型支持 FC | 解析失败率 ~10-15% |

实际链路：`_invoke_react()` 优先走 FunctionCallAgent，失败自动切换到 ReActAgent。

### 4. 反思闭环（重点）

```
① Evidence Collector：从执行步骤中提取 rag_search/web_search 的实际返回内容
② ReflectionAgent：LLM 对照 用户问题+回答+证据，打出 quality_score，判断是否幻觉
③ 三级响应：
   ≥80 → 直接返回
   60-79 → SelfCorrector 单次 LLM 修复（省 token）
   <60 → 携带反思反馈重新执行 Agent（最多 2 次）
```

为什么加 Evidence Collector：反思不能靠 LLM 凭空判断幻觉——需要把 Agent 实际拿到的工具返回内容作为对照基准。

### 5. 可观测性

每次请求记录：trace_id、执行步骤、thought/action/observation、tool_calls、reflection_score、degraded 状态。

---

## 六、四层记忆

| 层 | 存储 | 角色 | 生命周期 |
|------|------|------|------|
| L1 | 内存 dict | 最终兜底，始终可用 | 进程重启丢失 |
| L2 | Redis List | 热路径加速，滑动窗口 + LLM 摘要压缩 | 可配置 TTL |
| L3 | PostgreSQL | 持久化真相源 | 永久 |
| L4 | Milvus | 语义向量召回（历史相似记忆） | 永久 |

读取：优先级链降级（Redis → PG → 内存），每层叠加 Milvus 语义召回。

写入：并行旁路，各层 try/except 隔离，失败不阻塞其他层。

已知问题：`_pg_ok` / `_redis_ok` 是单向熔断——故障一次后永久禁用直到重启，应改为三态断路器。

---

## 七、领域安全策略（可插拔 DomainPolicy）

底层平台通用，心血管 Demo 的医疗规则设计成可插拔的 `DomainPolicy`：

- 高风险关键词识别（"胸痛""呼吸困难"等）
- 急诊安全分流（直接返回 120 提示，短路后续 Agent 调用）
- 非诊断免责声明
- 禁止擅自停药、换药或调整剂量

通用底座和业务策略解耦，加新领域只需加一个配置实例。

---

## 八、ReAct、Plan-and-Execute 与 Function Calling 的关系（核心概念）

> 这是面试中最容易被追问的概念题。三个概念不在同一维度，不能简单理解为三个并列的 Agent 模式。

### 1. ReAct：思考与行动交替的执行模式

```text
Thought → Action → Observation → Thought → ... → Final Answer
```

ReAct 解决的核心问题：**每一步现在应该做什么**。适合工具调用顺序不固定、需要根据中间结果动态决定下一步的场景。

### 2. Plan-and-Execute：先规划，再执行

```text
Plan → Execute Step 1 → Execute Step 2 → ... → Replan（如需）→ Synthesize
```

Plan-and-Execute 解决的核心问题：**复杂任务整体应该拆成哪些步骤**。适合步骤较多、目标明确、需要展示清晰任务计划的场景。

### 3. Function Calling：工具调用的协议机制

Function Calling 不是推理策略，而是模型和程序之间约定工具调用格式的**协议**。模型返回结构化 JSON（工具名 + 参数），后端校验后执行实际函数，结果返回模型继续生成。

Function Calling 解决的核心问题：**模型如何以结构化、安全的方式调用程序工具**。

### 4. 四层层次关系

```text
┌────────────────────────────────────────┐
│ 任务编排层                              │
│ ReAct / Plan-and-Execute               │
│ 决定任务如何推进、是否拆解和是否重规划   │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│ 执行层                                  │
│ ReAct Loop                             │
│ LLM → 工具调用 → 观察结果 → LLM         │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│ 工具调用协议层                          │
│ Function Calling / 文本 Action 解析     │
│ 表达工具名、参数并接收工具结果           │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│ 具体工具层                              │
│ RAG / Web Search / Calculator           │
└────────────────────────────────────────┘
```

### 5. 本项目中的实际关系

```text
ModeRouter
├── react
│    └── _invoke_react()
│         ├── FunctionCallAgent（bind_tools，优先）
│         └── ReActAgent（正则解析，降级）
│
└── plan_execute
     ├── PlannerAgent
     ├── 逐个子任务调用 _invoke_react()
     │    ├── FunctionCallAgent
     │    └── ReActAgent
     └── Synthesizer
```

Function Calling 在 ReAct 执行中发挥作用，但它负责的是工具调用的表达和传输，不负责上层任务拆解。

### 6. 面试简洁回答

> ReAct、Plan-and-Execute 和 Function Calling 解决的是不同层次的问题。ReAct 是执行策略，通过 Thought、Action 和 Observation 的循环动态决定下一步；Plan-and-Execute 是任务规划策略，先把复杂任务拆成多个子任务，再逐步执行；Function Calling 是模型调用后端工具的结构化协议，负责把工具名和参数以 JSON 的形式传给程序。我的项目通过模式路由选择 ReAct 或 Plan-and-Execute，具体工具调用可以使用 Function Calling，RAG、搜索和计算器则作为底层工具被它们调用。

---

## 九、工程化能力

| 层 | 技术 | 用途 |
|------|------|------|
| API | FastAPI + Uvicorn | HTTP 服务 |
| Agent | LangChain FC + 正则解析版 | Agent 循环与工具调用 |
| LLM | OpenAI API 兼容 | 所有智能决策 |
| 向量库 | Milvus Lite + SentenceTransformers | 语义检索 + 长期记忆 |
| 缓存 | Redis | 短期记忆热路径 + 压缩 |
| 持久化 | PostgreSQL + SQLAlchemy + asyncpg | 会话全量存储 |
| 分词 | jieba | RAG BM25 中文关键词 |
| 前端 | React 18 + TS + Vite + Tailwind | UI |
| 文档解析 | pypdf + unstructured | PDF/TXT 摄入 |
| 测试 | pytest + pytest-asyncio | 异步 UT（93 passed） |

工程特性：SSE 流式输出、文件上传安全校验、路径穿越防护、Docker Compose、GitHub Actions CI、多层降级机制。

---

## 十、已知不足与改进方向

| 优先级 | 问题 | 方案 |
|------|------|------|
| P0 | 记忆层单向熔断 | `_pg_ok` 改为三态断路器，PG 恢复后自动回连 |
| P1 | 降级期数据可能丢失 | PG 不可用时写 WAL，恢复后回放 |
| P1 | 无流式输出 | SSE 词级推送 |
| P2 | 工具结果无缓存 | rag_search 短期缓存同 query |
| P2 | 评测体系缺失 | 自动评测：正确性/完整性/安全性 |

---

## 十一、后续规划

1. 将前端硬编码的领域配置改成可选择的领域 Profile
2. 增加更多领域策略和领域评测集
3. 完善 Agent、RAG、SSE 和服务降级的端到端测试
4. 进一步统一 Milvus Lite 和 Milvus Server 配置
5. 增加 pgvector 适配实现，验证 VectorStore 抽象
6. 引入 Alembic 做数据库迁移
7. 增加检索召回率、答案准确率、延迟和成本等评测指标

---

## 附：LangGraph 实现说明（面试时不主动提）

项目中有一份完整的 LangGraph ReAct Agent 实现（`langgraph_agent.py`，307 行），使用 `StateGraph` + `ToolNode` + `Evaluator` 节点 + 条件边构建，含质量分不达标自动重试逻辑。但这份实现未接入 orchestrator 和 agent_service 的服务链路，不在实际运行路径中。

保留的原因是：理解并对比"框架版"和自己写的版本的差异。面试时不必主动提及——如果被问到"用没用 LangGraph"，可以如实说"有一份完整的 StateGraph 实现用于学习对比，但当前运行链路用的是 LangChain 的 bind_tools 加自己写的编排逻辑"。
