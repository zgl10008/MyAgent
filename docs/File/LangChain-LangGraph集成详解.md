# LangChain + LangGraph 集成详解

> 本文档详细讲解项目中新增的 LangChain/LangGraph 集成，覆盖架构设计、代码逐行解析、三种 Agent 模式对比。

---

## 目录

- [1. 背景：为什么要加 LangChain/LangGraph](#1-背景为什么要加-langchainlanggraph)
- [2. 架构总览：三模式并存](#2-架构总览三模式并存)
- [3. 核心：LangGraph StateGraph ReAct Agent](#3-核心langgraph-stategraph-react-agent)
- [4. 编排器扩展：第三模式 langgraph](#4-编排器扩展第三模式-langgraph)
- [5. 服务层集成：工具包装 + 依赖注入](#5-服务层集成工具包装--依赖注入)
- [6. ETL 集成：LangChain 分块策略](#6-etl-集成langchain-分块策略)
- [7. 三种模式深度对比](#7-三种模式深度对比)
- [8. 面试高频追问](#8-面试高频追问)

---

## 1. 背景：为什么要加 LangChain/LangGraph

### 1.1 原有状态

项目最初完全从零自研 Agent 引擎，**`app/` 目录中 LangChain/LangGraph 零引用**：

```
自研内容：
├── ReAct 循环      → react_agent.py    (纯正则解析 Thought/Action/Final Answer)
├── Plan-and-Execute → planner.py        (LLM 生成计划 → 顺序执行 → 失败重规划)
├── Reflection 反思  → reflection.py     (独立 LLM 质检评分)
├── 工具系统         → tools/            (BaseTool + ToolRegistry + ToolRouter)
├── 意图识别         → intent/recognizer.py (LLM few-shot + 关键词降级)
└── 记忆系统         → memory/           (Redis + PostgreSQL + InMemory)
```

**但是在 `pyproject.toml` 和 `requirements.txt` 中已经声明了依赖**：

```
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-community>=0.3.0
langgraph>=0.3.0
```

### 1.2 添加动机

面试场景下，仅会自研显得不够全面。工业界 80% 的 Agent 项目使用 LangChain/LangGraph 框架。**同时掌握自研和框架**才是最佳定位：

| 能力维度 | 仅自研 | 仅框架 | **双模式（本项目）** |
|----------|--------|--------|---------------------|
| 底层理解 | ✅ | ❌ | ✅ |
| 工业标准 | ❌ | ✅ | ✅ |
| 面试竞争力 | 中等 | 中等 | **强** |

---

## 2. 架构总览：三模式并存

```
AgentService (顶层服务)
│
├── IntentRecognizer → 意图分类（LLM + 关键词降级）
│
└── AgentOrchestrator ─── 三种模式 ───┐
    │                                  │
    ├── mode="react" ────────── ReActAgent (自研)
    │   └── 正则解析 Thought/Action/Final Answer
    │   └── for 循环，步数上限 10
    │
    ├── mode="plan_execute" ── PlannerAgent (自研)
    │   └── LLM 生成 SubTask 列表
    │   └── 顺序执行 + 失败 replan (最多 2 次)
    │   └── 规划失败 → 降级 react
    │
    └── mode="langgraph" ───── LangGraphReActAgent (框架, 新增)
        └── StateGraph 状态机
        └── ToolNode 自动工具调用
        └── 失败 → 降级 react
```

**关键设计原则**：新模块是纯增量，原 `react` 和 `plan_execute` 一行代码未改。

### 改动文件清单

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `app/core/agent/langgraph_agent.py` | **新建** | 161 行 | LangGraph ReAct Agent |
| `app/core/agent/orchestrator.py` | 改 3 处 | +40 行 | 新增 langgraph 模式分支 |
| `app/core/agent_service.py` | 改 2 处 | +30 行 | 工具包装 + Agent 注入 |
| `app/etl/chunker.py` | 改 2 处 | +12 行 | LangChain 分块策略 |
| `app/models/schemas.py` | 改 1 处 | 1 行 | agent_mode 描述更新 |

---

## 3. 核心：LangGraph StateGraph ReAct Agent

### 3.1 文件位置

`app/core/agent/langgraph_agent.py`（161 行）

### 3.2 逐块解析

#### 3.2.1 State 定义（图的状态容器）

```python
class AgentState(TypedDict):
    """LangGraph ReAct 状态：消息列表累加。"""
    messages: Annotated[list[BaseMessage], add_messages]
```

**知识点**：
- `TypedDict` 是 LangGraph 的核心概念，定义了状态图的数据结构
- `Annotated[list, add_messages]` 是关键注解——`add_messages` 是 LangGraph 内置的 reducer 函数，新消息自动追加到列表，不会覆盖旧消息。这意味着每次节点返回 `{"messages": [new_msg]}` 时，new_msg 会被 append 而不是 replace
- 对比自研版：自研版用 `history_lines: List[str]` 手动拼接，LangGraph 用消息列表自动累积

#### 3.2.2 构建图（`_build_graph` 方法）

```python
def _build_graph(self):
    tool_node = ToolNode(self._tools)        # LangGraph 内置工具执行节点
    llm_with_tools = self._llm.bind_tools(self._tools)  # 绑定工具到 LLM

    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"              # 还有工具要调
        return END                       # 生成最终答案，结束

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")                     # 入口 → agent
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")                   # 工具完成后回到 agent

    return workflow.compile()
```

**图结构可视化**：

```
                              ┌─────────────────┐
                              │     START        │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                         ┌───→│     agent       │
                         │    │  (LLM 推理)      │
                         │    └────────┬────────┘
                         │             │
                         │    should_continue?
                         │    ┌────┴────┐
                         │    │         │
                         │ tool_calls?  no tool_calls
                         │    │         │
                         │    ▼         ▼
                         │ ┌──────┐  ┌─────┐
                         └─┤ tools │  │ END │
                           └──────┘  └─────┘
```

**与自研版的关键差异**：

| 维度 | 自研 react_agent.py | LangGraph langgraph_agent.py |
|------|---------------------|------------------------------|
| 控制流 | `for step_idx in range(max_steps)` | `StateGraph` 状态机 |
| 工具调用解析 | `re.search(r"Action:\s*(\S+)")` | `last.tool_calls` (结构化) |
| 工具执行 | 手动 `await self._tools.invoke(action, args)` | `ToolNode` 自动执行 |
| 状态管理 | `history_lines: List[str]` 逐条拼接 | `add_messages` reducer 自动累加 |
| 检查点/回滚 | 无 | `MemorySaver`（可替换为 PostgresSaver） |
| 模型接口 | `acomplete(messages) -> str` | `ChatOpenAI.bind_tools(tools)` |

#### 3.2.3 System Prompt

LangGraph 版用简化的系统提示，因为 `ChatOpenAI.bind_tools()` 会自动注入工具的函数签名（function calling），模型通过原生 tool_calls 机制识别工具，不需要正则解析。

```python
REACT_SYSTEM_PROMPT = """你是一个严谨的智能助手，必须使用 ReAct（推理+行动）方式回答问题。

## 规则
- 需要外部信息时调用可用工具；不需要时可直接回答。
- 每次工具调用后结合 Observation 判断是否已有足够信息给出最终答案。
- 不要编造工具名称或 Observation。
- 使用中文回答。
"""
```

#### 3.2.4 步骤提取（`_collect_steps`）

LangGraph 的消息列表是 LangChain `BaseMessage` 对象，需要转换回 dict 供前端展示：

```python
def _collect_steps(self, messages: list[BaseMessage]) -> list[dict]:
    steps = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            rec = {"step": step_idx, "phase": "react", "thought": msg.content or ""}
            if msg.tool_calls:
                rec["action"] = msg.tool_calls[0].get("name", "")
                rec["action_input"] = msg.tool_calls[0].get("args", {})
            steps.append(rec)
        elif isinstance(msg, ToolMessage):
            if steps:
                steps[-1]["observation"] = str(msg.content)[:2000]
    return steps
```

#### 3.2.5 异步执行入口（`run` 方法）

```python
async def run(self, query: str, context: Dict[str, Any]) -> AgentResult:
    tool_descriptions = str(context.get("tool_descriptions", ""))
    extra_system = str(context.get("extra_system", ""))
    system_text = _build_system_prompt(tool_descriptions, extra_system)

    initial_messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=query),
    ]

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": self._max_steps * 2 + 1,  # 每次循环 = agent + tools 两步
    }

    result = await self._graph.ainvoke({"messages": initial_messages}, config)
    # 提取最终答案...
```

**关键技术点**：
- `recursion_limit`：LangGraph 的图递归上限。每个 ReAct 循环 = agent → tools → agent（2 次图遍历），所以设置 `max_steps * 2 + 1`
- `thread_id`：LangGraph 的会话标识，用于多轮对话的状态隔离（配合 MemorySaver 实现检查点）
- `ainvoke`：LangGraph 的异步调用方法

### 3.3 LangGraph 版使用的核心 LangChain/LangGraph 组件

```python
from langchain_openai import ChatOpenAI              # OpenAI 兼容的 Chat Model
from langchain_core.messages import (                # 消息类型体系
    BaseMessage, HumanMessage, SystemMessage,
    AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END   # 状态图核心
from langgraph.graph.message import add_messages     # 消息累加 reducer
from langgraph.prebuilt import ToolNode              # 内置工具执行节点
```

---

## 4. 编排器扩展：第三模式 langgraph

### 4.1 OrchestrationMode 枚举扩展

```python
# 原代码（orchestrator.py 第 20 行）
OrchestrationMode = Literal["react", "plan_execute"]

# 改为
OrchestrationMode = Literal["react", "plan_execute", "langgraph"]
```

### 4.2 LangGraph Agent 注入

编排器通过 **setter 注入** 接收 LangGraph agent：

```python
class AgentOrchestrator:
    def __init__(self, ...):
        ...
        self._langgraph_agent: Optional[LangGraphReActAgent] = None  # 初始为空

    def set_langgraph_agent(self, agent: LangGraphReActAgent) -> None:
        """注入 LangGraph Agent 实例（由 AgentService 在初始化时调用）。"""
        self._langgraph_agent = agent
```

**为什么用 setter 注入而不是构造函数参数？**
编排器通过 Protocol 实现依赖反转，构造函数参数全是 Protocol 类型。LangGraphReActAgent 是具体类，通过 setter 注入可以避免破坏协议化设计，同时保持编排器对 LangGraph 的依赖是可选的。

### 4.3 run() 方法新增分支

```python
elif mode == "langgraph":
    mode_used = "langgraph"
    result = await self._run_langgraph(user_input, session_id, intent_ctx, trace_id, steps)
    # LangGraph 失败也支持降级到 react
    if not result.success and self._fallback_on_plan_failure:
        degraded = True
        result = await self._run_react(..., suffix="[LangGraph 降级] ")
        mode_used = "react"
```

### 4.4 _run_langgraph 方法

```python
async def _run_langgraph(self, user_input, session_id, intent, trace_id, steps):
    if self._langgraph_agent is None:
        return AgentResult(success=False, error="LangGraph Agent 未注入")

    ctx = {
        "session_id": session_id,
        "tool_descriptions": getattr(self._tools, "get_tools_description", lambda: "")(),
        "extra_system": f"当前识别意图：{intent.intent}，槽位：{intent.slots}",
    }
    result = await self._langgraph_agent.run(user_input, ctx)
    steps.extend(result.steps)
    return AgentResult(success=result.success, final_answer=result.final_answer, ...)
```

---

## 5. 服务层集成：工具包装 + 依赖注入

### 5.1 项目工具 → LangChain StructuredTool

LangGraph 的 `ToolNode` 需要 LangChain 格式的工具对象。项目原有的工具是自研 `BaseTool` 体系。解决方案：用 `StructuredTool.from_function()` 做适配：

```python
from langchain_core.tools import StructuredTool

_lg_tools = []
for t in self._tool_registry.get_all_tools():      # 遍历项目工具
    def _make_func(tool):
        async def _fn(**kwargs: Any) -> str:       # 代理到 tool.execute()
            return await tool.execute(**kwargs)
        _fn.__name__ = tool.name
        return _fn

    lg_tool = StructuredTool.from_function(
        coroutine=_make_func(t),                    # 异步函数
        name=t.name,                                # 工具名
        description=t.description,                  # 工具描述（LLM 据此判断何时使用）
        args_schema=None,                           # 从函数签名自动推断参数 schema
    )
    _lg_tools.append(lg_tool)
```

### 5.2 ChatOpenAI 创建

```python
self._langgraph_agent = LangGraphReActAgent(
    model_name=settings.openai_model,         # "deepseek-v4-flash"
    api_key=settings.openai_api_key,          # DeepSeek API Key
    base_url=settings.openai_api_base,        # "https://api.deepseek.com"
    tools=_lg_tools,
    max_steps=int(config.get("react_max_steps", 10)),
)
```

**为什么 DeepSeek 可以用 ChatOpenAI？**
DeepSeek API 与 OpenAI API 完全兼容（相同的 `/v1/chat/completions` 端点和请求/响应格式）。`langchain_openai.ChatOpenAI` 通过 `base_url` 参数可以指向任何 OpenAI 兼容的 API。

### 5.3 注入到编排器

```python
self._orchestrator.set_langgraph_agent(self._langgraph_agent)
logger.info("LangGraph Agent 已注入编排器（langgraph 模式可用）")
```

整个过程用 try/except 包裹，LangGraph 创建失败不影响其他模式：
```python
try:
    # ... 创建 LangGraph agent ...
except Exception as exc:
    logger.warning("LangGraph Agent 创建失败，langgraph 模式不可用: {}", exc)
```

---

## 6. ETL 集成：LangChain 分块策略

### 6.1 新增策略枚举

```python
class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    PARAGRAPH = "paragraph"
    LANGCHAIN = "langchain"    # ← 新增
```

### 6.2 实现

```python
def _chunk_langchain(self, text: str) -> List[str]:
    """使用 LangChain RecursiveCharacterTextSplitter 分块。"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=self.chunk_size,        # 512
        chunk_overlap=self.chunk_overlap,  # 64
        separators=["\n\n", "\n", "。", ". ", " "],
    )
    return splitter.split_text(text)
```

**对比原有递归分块**：

| 维度 | 自研 _chunk_recursive | LangChain RecursiveCharacterTextSplitter |
|------|----------------------|------------------------------------------|
| 分隔符优先级 | 手动递归试探 | 内置优先级队列 |
| 边界处理 | 自定义合并逻辑 | 成熟的 seek 策略 |
| token 计数 | tiktoken 独立封装 | 可选 tiktoken 集成 |
| 生产可靠性 | 适合演示 | 工业级（被数千项目验证） |

---

## 7. 三种模式深度对比

### 7.1 一句话总结

| 模式 | 一句话 | 适合场景 |
|------|--------|----------|
| `react` | 自研正则解析 ReAct 循环 | 需要完全控制、定制化输出格式 |
| `plan_execute` | 自研先规划再执行 | 复杂多步任务，需要整体规划 |
| `langgraph` | LangGraph 官方状态机 | 工业标准，需要检查点/人机协同 |

### 7.2 技术维度对比

| 维度 | react (自研) | plan_execute (自研) | langgraph (框架) |
|------|-------------|---------------------|-------------------|
| **Agent 控制流** | for 循环 + 正则 | LLM 计划 → 遍历 → replan | StateGraph 状态机 |
| **工具调用** | 文本解析 Action/Action Input | 文本解析 | Function Calling (结构化) |
| **状态持久化** | 无（依赖外部记忆） | 无 | MemorySaver 检查点 |
| **人机协同** | 不支持 | 不支持 | `interrupt()` 原生支持 |
| **流式输出** | 需自实现 | 需自实现 | `astream_events()` 原生支持 |
| **学习成本** | 需读源码 | 需读源码 | 需学 LangGraph API |
| **代码行数** | 291 行 | ~150 行 | 161 行 |
| **面试展示** | 底层能力 | 规划能力 | 框架能力 |

### 7.3 失败降级链路

```
用户请求
  │
  ├── langgraph 模式
  │     ├── 成功 → 返回结果
  │     └── 失败 → 降级 react 模式 → 返回结果（标记 degraded=true）
  │
  ├── plan_execute 模式
  │     ├── 成功 → 返回结果
  │     └── 失败 → 降级 react 模式 → 返回结果（标记 degraded=true）
  │
  └── react 模式（最终兜底）
        ├── 成功 → 返回结果
        └── 失败 → 返回错误信息
```

---

## 8. 面试高频追问

### Q1: 为什么同时保留自研和 LangGraph 两个版本？

**答**：展示两个层次的能力：
1. **自研 React**：证明理解 Agent 底层循环机制（正则解析、步数控制、Observation 拼接），不是只会调包
2. **LangGraph**：证明掌握工业标准框架（StateGraph、ToolNode、Function Calling），能直接上手企业项目

### Q2: LangGraph 相比自研 React 有什么优势？

**答**：
1. **结构化工具调用**：Function Calling 比文本解析更可靠（不会有 JSON 解析失败）
2. **检查点/持久化**：MemorySaver 可存储任意步骤的状态，支持断点续跑和回滚
3. **人机协同**：`interrupt()` 可在任意节点暂停，等待人工审批后继续
4. **生态系统**：`astream_events()` 原生流式、`LangSmith` 调试追踪、社区工具链
5. **复杂工作流**：支持并行分支、子图嵌套、map-reduce 模式

### Q3: LangGraph 相比自研 React 有什么劣势？

**答**：
1. **黑盒程度更高**：ToolNode 内部实现不可见，调试需依赖 LangSmith
2. **依赖体积大**：langchain + langgraph 包体较大，冷启动慢
3. **版本兼容风险**：0.x 阶段 API 变动频繁
4. **定制局限性**：对非标准工具调用格式（如 Thought/Observation 文本拼接）支持不如自研灵活

### Q4: 三种模式的生产选型建议？

**答**：
- **快速原型 / 面试演示** → 全用，展示技术广度
- **内部工具 / 简单任务** → react 自研版（依赖少，启动快）
- **复杂多步任务** → plan_execute 或 langgraph（规划 + 降级）
- **需要人机协同 / 审批** → langgraph（`interrupt()` 原生支持）
- **企业级生产环境** → langgraph（社区支持、生态完善、可观测性好）

### Q5: 项目的工具是如何适配 LangChain 的？

**答**：使用 `StructuredTool.from_function()`，核心是闭包代理模式：

```python
for t in self._tool_registry.get_all_tools():
    def _make_func(tool):
        async def _fn(**kwargs):
            return await tool.execute(**kwargs)  # 代理到项目原有 BaseTool.execute()
        return _fn
    StructuredTool.from_function(coroutine=_make_func(t), name=t.name, description=t.description)
```

优势：不改原有工具代码一行，通过函数包装实现适配。如果项目工具改为继承 LangChain 的 `BaseTool`，适配代码可直接删除。

### Q6: 为什么要增加 LangChain 分块策略？

**答**：在简历上列出 `LangChain` 需要有实际代码引用。`RecursiveCharacterTextSplitter` 是 LangChain 中使用最广泛的文本处理工具，集成到 ETL 流水线中既实用又能证明 LangChain 的使用经验。

---

## 附录：完整文件清单

```
app/
├── core/
│   ├── agent/
│   │   ├── langgraph_agent.py    ← 新建（161行）
│   │   ├── orchestrator.py       ← 已改（+40行）
│   │   ├── react_agent.py        ← 未改
│   │   ├── planner.py            ← 未改
│   │   └── reflection.py         ← 未改
│   ├── agent_service.py          ← 已改（+30行）
│   └── ...
├── etl/
│   └── chunker.py                ← 已改（+12行）
└── models/
    └── schemas.py                ← 已改（1行）
```
