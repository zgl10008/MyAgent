# 项目介绍：通用 Agent/RAG 平台

## 一、项目定位

这是一个**通用 Agent/RAG 平台**，当前以“**心血管慢病管理**”作为默认 Demo，同时支持上传任意领域的 PDF、Markdown 和文本文件，自动构建对应知识库并进行问答。

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

### 3. Agent 模式

Agent 根据用户问题进行意图识别和模式路由，然后自主选择工具。目前主要包含：

- RAG 文档检索；
- Web Search；
- Calculator；
- 多模型路由；
- ReAct；
- Plan-and-Execute；
- Function Calling。

因此，RAG 在系统中既是一个**独立的问答能力**，也可以作为 Agent 的一个**可调用工具**。

---

## 三、一次 Agent 请求的处理流程

```text
用户请求
  ↓
请求校验与领域策略识别
  ↓
高风险问题安全分流
  ↓
意图识别
  ↓
Agent 模式路由
  ├── ReAct
  │    └── ReAct 执行循环
  │         └── Function Calling 或文本 Action 解析
  │              └── 调用 RAG、搜索或计算工具
  │
  └── Plan-and-Execute
       ├── Planner：拆解任务
       ├── Executor：每个子任务复用 ReAct 执行循环
       │    └── Function Calling 或文本 Action 解析
       ├── Replanner：失败后重新规划
       └── Synthesizer：汇总结果
  ↓
反思与质量评分
  ↓
返回最终答案和执行轨迹
```

例如用户提出：

> 根据我上传的指南，解释某个指标，并结合最新资料给出分析。

Agent 可以先调用 RAG 检索本地文档，再调用 Web Search 获取实时信息，最后综合多个工具的结果生成回答。前端可以展示 Agent 步骤、工具调用和检索引用，方便调试和演示。

---

## 四、RAG 实现

### 1. 文档处理流程

```text
文件上传
  ↓
格式与安全校验
  ↓
文本解析
  ↓
文档分块
  ↓
Embedding
  ↓
写入向量库和数据库
```

目前支持 PDF、TXT、MD 和 Markdown 文件。

### 2. 混合检索

系统同时支持：

- 向量语义检索；
- BM25 关键词检索；
- 混合检索；
- RRF 倒数排名融合；
- 可选的重排序能力。

这样可以兼顾语义相近的问题，以及对专业名词、指标名和关键词敏感的问题。

### 3. 向量存储抽象

当前主要使用 Milvus Lite 保存向量数据，同时定义了 `VectorStore` 抽象协议。RAG 业务层不直接依赖具体 Milvus 类型，未来可以替换为：

- pgvector；
- Milvus Server；
- 其他向量数据库。

### 4. 失败降级

```text
sentence-transformers
  ↓ 失败
字符 n-gram Embedding
  ↓
Milvus Lite
  ↓ 失败
内存向量存储
```

即使本地没有完整启动 Milvus，也可以使用内存实现完成基本检索。

---

## 五、Agent 实现

### 1. 意图识别

系统会判断用户请求属于：

- 知识问答；
- 实时搜索；
- 数学计算；
- 文档管理；
- 普通闲聊。

意图识别同时支持 LLM 和规则降级。LLM 判断失败时，系统回退到规则识别，避免单点失败。

### 2. 模式路由

根据问题复杂度选择不同的任务编排方式，并在执行阶段选择具体的工具调用后端：

- 简单任务可以直接进入 ReAct 执行循环；
- 多步骤、目标明确的复杂任务使用 Plan-and-Execute；
- ReAct 的每一轮工具调用可以使用 Function Calling，也可以使用文本 Action 解析；
- RAG、Web Search 和 Calculator 是具体工具，而不是上层 Agent 模式。

因此，Function Calling 不与 ReAct、Plan-and-Execute 并列为第三种任务编排模式。

### 3. 工具调用

工具通过统一的工具注册机制接入 Agent。工具返回结果后，Agent 会继续分析，而不是简单地把工具输出直接当成最终答案。

### 4. 可观测性

每次 Agent 请求都会记录：

- `trace_id`；
- Agent 执行步骤；
- thought；
- action；
- action input；
- observation；
- final answer；
- tool calls；
- reflection score；
- degraded 状态。

这样可以定位问题出在意图识别、工具选择、工具执行、检索结果还是最终答案生成。

---

## 六、记忆与存储设计

### PostgreSQL

负责持久化数据：

- 用户；
- 会话；
- 消息；
- 文档元数据；
- 文档块；
- 长期记忆。

### Redis

负责短期和高频数据：

- 当前会话记忆；
- 滑动窗口；
- TTL；
- 热点缓存；
- 摘要缓存。

### Milvus Lite

负责：

- 文档向量检索；
- 长期语义记忆；
- 相似度搜索。

### 内存

主要用于本地开发、测试，以及 Redis、PostgreSQL 或 Milvus 不可用时的降级。

核心原则是：关系数据、短期状态和语义数据分别使用适合的存储，而不是让一个数据库承担所有职责。

---

## 七、医疗领域安全策略

底层平台是通用的，但保留心血管慢病管理作为默认 Demo，并把医疗相关规则设计成可插拔的 `DomainPolicy`。

目前有两个领域策略：

- `general`；
- `cardiovascular`。

心血管策略包括：

- 疾病相关系统提示；
- 高风险关键词识别；
- 急诊安全提示；
- 非诊断免责声明；
- 禁止擅自停药、换药或调整剂量。

例如用户输入“突然胸痛、呼吸困难”，系统会在 Agent、RAG、普通聊天等入口提前进行安全分流，直接提示及时就医，而不是继续调用模型生成可能有风险的回答。

通用 Agent 底座和业务策略解耦，未来可以增加教育、法律或企业知识库等领域，而不需要修改核心 Agent 逻辑。

---

## 八、ReAct、Plan-and-Execute 与 Function Calling 的关系

这三个概念不处于同一个维度，不能简单理解为三个完全并列、互相替代的 Agent。

### 1. ReAct：思考与行动交替的执行模式

ReAct 的核心是：

```text
Thought → Action → Observation → Thought → ... → Final Answer
```

含义是：

1. Agent 分析当前问题；
2. 决定调用哪个工具；
3. 获得工具返回结果；
4. 根据结果继续分析；
5. 必要时再次调用工具；
6. 最后生成答案。

例如：

```text
Thought: 用户的问题需要查询上传文档
Action: 调用 RAGSearchTool
Observation: 返回相关文档片段
Thought: 文档信息足够，可以组织答案
Final Answer: 结合引用生成回答
```

ReAct 解决的主要问题是：**每一步现在应该做什么**。

它适合：

- 工具调用顺序不固定；
- 需要根据中间结果动态决定下一步；
- 搜索、RAG、计算等工具需要灵活组合的任务。

### 2. Plan-and-Execute：先规划，再执行

Plan-and-Execute 将复杂任务拆成两个阶段：

```text
Plan → Execute Step 1 → Execute Step 2 → ... → Synthesize
```

例如用户要求：

> 结合上传的指南、最新公开资料和计算结果，分析某项健康管理方案。

系统可以先规划：

```text
1. 从上传文档中检索相关指南
2. 查询最新公开资料
3. 计算或比较关键指标
4. 综合结果并给出引用
```

然后逐步执行这些计划，最后汇总结果。

Plan-and-Execute 解决的主要问题是：**复杂任务整体应该拆成哪些步骤**。

它适合：

- 任务步骤较多；
- 目标比较明确；
- 需要展示清晰任务计划；
- 希望提升复杂任务执行的可控性和可解释性。

### 3. Function Calling：工具调用的协议机制

Function Calling 不是一种完整的推理策略，而是模型和程序之间约定工具调用格式的机制。

模型不直接执行函数，而是返回结构化调用信息，例如：

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "高血压管理目标",
    "top_k": 5
  }
}
```

后端收到后：

1. 根据工具名查找已注册工具；
2. 校验参数；
3. 执行实际 Python 函数；
4. 把工具结果返回给模型；
5. 模型继续生成最终答案或发起下一次工具调用。

Function Calling 解决的主要问题是：**模型如何以结构化、安全的方式调用程序工具**。

它适合：

- 工具参数结构明确；
- 需要严格参数校验；
- 需要减少解析自然语言 Action 的不确定性；
- 需要让工具调用更容易审计和观测。

### 4. 三者的层次关系

这三个概念应当放在不同层次理解：

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

更准确地说：

- Plan-and-Execute 负责复杂任务的拆解、执行顺序、失败重规划和结果汇总；
- ReAct 负责在执行过程中根据当前观察动态决定下一步；
- Function Calling 负责把工具调用结构化为工具名和 JSON 参数；
- 文本 Action 解析是 Function Calling 的另一种兼容实现方式；
- RAG、Web Search 和 Calculator 是被执行器调用的具体工具。

推荐的组合方式是：

```text
Plan-and-Execute
  ├── Planner：生成任务计划
  ├── Executor：逐个执行子任务
  │    └── ReAct Loop
  │         └── Function Calling 或文本 Action 解析
  ├── Replanner：失败后重新规划
  └── Synthesizer：汇总结果
```

简单任务则可以直接使用：

```text
ReAct Loop
  └── Function Calling
       ├── RAGSearchTool
       ├── WebSearchTool
       └── CalculatorTool
```

### 5. 本项目中的实际关系

本项目的上层模式路由只负责选择两种任务编排范式：

```text
用户请求
  ↓
意图识别与复杂度判断
  ↓
ModeRouter
  ├── react
  │    └── _invoke_react()
  │         ├── FunctionCallAgent
  │         └── ReActAgent
  │
  └── plan_execute
       ├── PlannerAgent
       ├── 逐个子任务调用 _invoke_react()
       │    ├── FunctionCallAgent
       │    └── ReActAgent
       └── 汇总结果
```

其中：

- `react` 和 `plan_execute` 是上层任务编排模式；
- `FunctionCallAgent` 和 `ReActAgent` 是 ReAct 执行循环的两种实现；
- `FunctionCallAgent` 使用 `bind_tools()` 和结构化 `tool_calls`；
- `ReActAgent` 使用文本形式的 `Thought / Action / Action Input` 协议；
- `RAGSearchTool`、`WebSearchTool` 和 `CalculatorTool` 是底层具体工具。

因此，当前项目不是：

```text
ModeRouter
├── react
├── plan_execute
└── function_calling
```

而是：

```text
ModeRouter
├── react
└── plan_execute
     └── ReAct 执行器
          ├── Function Calling 后端
          └── 文本协议后端
```

`AgentOrchestrator` 中的 `_invoke_react()` 会根据配置选择具体后端。开启 `use_function_calling` 且注入 `FunctionCallAgent` 时，普通 ReAct 请求和 Plan-and-Execute 的子任务都会使用 Function Calling 后端；否则使用传统文本协议版 `ReActAgent`。

因此，Function Calling 确实在 ReAct 执行中发挥作用，但它负责的是工具调用的表达和传输，不负责上层任务拆解。
### 6. 面试中的简洁回答

面试官如果问三者有什么区别，可以这样回答：

> ReAct、Plan-and-Execute 和 Function Calling 解决的是不同层次的问题。ReAct 是执行策略，通过 Thought、Action 和 Observation 的循环动态决定下一步；Plan-and-Execute 是任务规划策略，先把复杂任务拆成多个子任务，再逐步执行；Function Calling 是模型调用后端工具的结构化协议，负责把工具名和参数以 JSON 的形式传给程序。我的项目通过模式路由选择 ReAct 或 Plan-and-Execute，具体工具调用可以使用 Function Calling，RAG、搜索和计算器则作为底层工具被它们调用。

---

## 九、工程化能力

项目后端使用：

- Python 3.11+；
- FastAPI；
- Pydantic v2；
- SQLAlchemy Async；
- PostgreSQL；
- Redis；
- Milvus Lite；
- LangChain / LangGraph；
- OpenAI-compatible API。

前端使用：

- React；
- TypeScript；
- Vite；
- Tailwind CSS。

工程上还实现了：

- SSE 流式输出；
- 文件上传安全校验；
- 路径穿越防护；
- 扩展名校验；
- 文件大小限制；
- 空文件校验；
- Docker Compose；
- GitHub Actions CI；
- pytest；
- 多层降级机制。

当前在 Conda 的 `MyAgent` 环境中，完整 pytest 已通过：

```text
93 passed
```

前端 TypeScript 检查和生产构建也已通过。

---

## 十、项目核心亮点

### 1. Agent 不是单一模式

支持意图识别、模式路由、ReAct、计划执行和 Function Calling，能够展示完整的 Agent 编排能力。

### 2. RAG 不只是向量搜索

通过 BM25、向量搜索、RRF 和重排序，系统可以更好地处理中文文档和专业术语。

### 3. 考虑真实工程问题

系统针对外部服务不可用、模型调用失败、向量库不可用、Redis 不可用、上传风险文件和工具调用失败等情况提供了降级或错误处理。

### 4. 通用底座和领域策略解耦

系统不是边界不清晰的“万能 Agent”，而是通用平台、默认 Demo 和可插拔策略的组合。

---

## 十一、后续规划

1. 将前端硬编码的领域配置改成可选择的领域 Profile；
2. 增加更多领域策略和领域评测集；
3. 完善 Agent、RAG、SSE 和服务降级的端到端测试；
4. 进一步统一 Milvus Lite 和 Milvus Server 配置；
5. 增加 pgvector 适配实现，验证 VectorStore 抽象；
6. 引入 Alembic 做数据库迁移；
7. 增加检索召回率、答案准确率、延迟和成本等评测指标。

---

## 十二、一句话总结

> 这是一个以心血管慢病管理为默认 Demo，具备混合 RAG、可编排 Agent、分层记忆、故障降级和领域安全策略的通用 Agent/RAG 平台。
