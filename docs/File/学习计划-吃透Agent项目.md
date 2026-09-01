# AI Agent 项目学习计划 — 从零吃透到面试通关

> 适用对象：Agent 开发新手，目标是通过此项目掌握 Agent 核心原理，应对技术面试。

---

## 目录

1. [项目概览](#1-项目概览)
2. [学习路线图（5 个阶段）](#2-学习路线图5-个阶段)
3. [模块详解](#3-模块详解)
   - [3.1 Agent 核心循环](#31-agent-核心循环)
   - [3.2 RAG 全链路](#32-rag-全链路)
   - [3.3 记忆系统](#33-记忆系统)
   - [3.4 工具系统](#34-工具系统)
   - [3.5 模型路由与韧性](#35-模型路由与韧性)
   - [3.6 意图识别](#36-意图识别)
   - [3.7 反思质检](#37-反思质检)
   - [3.8 ETL 文档处理](#38-etl-文档处理)
   - [3.9 基础设施](#39-基础设施)
   - [3.10 前端可视化](#310-前端可视化)
4. [面试必备补充知识](#4-面试必备补充知识)
5. [面试话术模板](#5-面试话术模板)

---

## 1. 项目概览

### 一句话描述

背景调整一下。
> 一个企业级 AI Agent 智能客服平台，支持 ReAct 和 Plan-Execute 双模式、混合检索 RAG、三级记忆系统、多模型路由与熔断降级。

### 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI (异步) |
| Agent 引擎 | 自研 ReAct + Plan-and-Execute |
| RAG | Milvus Lite + BM25 自研 + RRF 融合 |
| 记忆 | Redis (短期) + PostgreSQL (持久化) + 内存 (降级) |
| LLM | OpenAI 兼容 API，多模型路由 |
| 向量模型 | Sentence-Transformers `all-MiniLM-L6-v2` (384 维) |
| 前端 | React + TypeScript + Vite |
| 部署 | Docker Compose (PostgreSQL 16 + Redis 7) |

### 架构图

```
用户请求 (HTTP)
    │
    ▼
┌──────────────┐
│   FastAPI     │  ← CORS / 路由 / 生命周期管理
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  意图识别     │  ← 关键词规则匹配 → 置信度 → 意图分类
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Agent 编排器 │  ← 选模式 / 调工具 / 管记忆 / 记录 Trace
└──┬───────┬───┘
   │       │
   ▼       ▼
┌─────┐ ┌──────────┐
│ReAct│ │Plan-Exec │  ← 两种执行模式，失败自动降级
└──┬──┘ └────┬─────┘
   │         │
   └────┬────┘
        │
        ▼
┌──────────────┐
│  反思质检     │  ← 质量评分 / 幻觉检测 / 完整性检查
└──────────────┘

支撑层：
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ RAG 引擎  │ │ 记忆系统  │ │ 工具系统  │ │ 模型路由  │
│ 向量+BM25 │ │ 三级存储  │ │ 注册表   │ │ 熔断降级  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 代码规模

```
app/
├── api/routes/          # 4 个路由文件
├── core/
│   ├── agent/           # 4 个 Agent 文件 (核心)
│   ├── memory/          # 3 个记忆文件
│   ├── rag/             # 3 个 RAG 组件
│   ├── tools/           # 5 个工具文件
│   ├── intent/          # 1 个意图识别
│   ├── rag_service.py   # RAG 统一服务 (465 行)
│   ├── persistent_memory.py  # 记忆统一管理 (298 行)
│   ├── agent_service.py      # Agent 服务层
│   └── session_repository.py # 会话持久化
├── infrastructure/
│   ├── llm/             # 熔断器 + 模型路由
│   ├── database/        # PostgreSQL ORM
│   ├── cache/           # Redis 缓存
│   ├── vectordb/        # Milvus 客户端
│   └── trace/           # 追踪系统
├── etl/                 # 文档解析→分块→入库
└── models/              # Pydantic 模型 + 枚举
frontend/
└── src/
    ├── components/      # 10 个 React 组件
    ├── pages/           # 4 个页面
    └── api/             # 4 个 API 客户端
```

---

## 2. 学习路线图（5 个阶段）

### 阶段一：环境搭建 + 跑起来（第 1 天）

**目标**：让项目在本地运行，看到效果。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（必填）
# 其他中间件（PG/Redis/Milvus）可选，项目能自动降级到内存模式

# 3. 启动后端
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. （可选）启动前端
cd frontend && npm install && npm run dev
```

**验收标准**：
- [ ] 访问 `http://localhost:8000/api/v1/health` 返回 `{"status":"ok"}`
- [ ] 能通过 `/api/v1/chat` 发送消息并得到回复
- [ ] 上传一个 PDF 文件，能在对话中检索到相关内容

**关键理解**：
- `.env` 文件的 `OPENAI_API_BASE` 可以指向任何 OpenAI 兼容 API（如国内中转）
- 没有 PostgreSQL/Redis 时，项目自动降级为内存模式，不影响学习

---

### 阶段二：追踪一条请求的完整链路（第 2-3 天）

**目标**：理解从用户发消息到收到回复，代码经过了多少层。

从 `POST /api/v1/chat` 入口开始，逐层阅读：

```
app/api/routes/chat.py          # ← 入口：接收消息
    │
    ▼
app/core/agent_service.py       # ← 服务层：组装 Agent
    │
    ▼
app/core/agent/orchestrator.py  # ← 编排器：选择 ReAct 还是 Plan-Execute
    │
    ├──► app/core/agent/react_agent.py    # ← ReAct 循环
    │    ├── Thought: 模型思考
    │    ├── Action: 调用工具
    │    └── Observation: 观察结果 → 循环
    │
    ├──► app/core/agent/planner.py        # ← Plan-Execute
    │    ├── plan(): LLM 生成子任务列表
    │    ├── execute(): 逐个执行子任务
    │    └── replan(): 失败重规划
    │
    ├──► app/core/agent/reflection.py     # ← 反思质检
    │    └── reflect(): 质量评分 + 幻觉检测
    │
    └──► app/core/tools/registry.py       # ← 工具调用
         ├── calculator.py  (安全计算)
         ├── search.py      (网络搜索)
         └── database.py    (SQL 查询)
```

**动手练习**：在 `react_agent.py` 的每一步打 `print`，观察 Thought → Action → Observation 的完整循环。

---

### 阶段三：逐个模块吃透（第 4-10 天）

每天 1-2 个模块，顺序按重要性排列。每个模块的学习方法：

1. **先读代码**（30 分钟）
2. **画流程图**（20 分钟）— 用自己的话画出数据流
3. **准备面试回答**（10 分钟）— 用 STAR 法则组织语言

详细解读见 [第 3 节](#3-模块详解)。

| 天数 | 模块 | 重要性 | 面试频率 |
|------|------|--------|---------|
| Day 4 | ReAct Agent | ⭐⭐⭐ | 必问 |
| Day 5 | Plan-Execute + 编排器 | ⭐⭐⭐ | 高频 |
| Day 6 | RAG 全链路 | ⭐⭐⭐ | 必问 |
| Day 7 | 记忆系统 | ⭐⭐ | 高频 |
| Day 8 | 模型路由 + 熔断器 | ⭐⭐ | 常问 |
| Day 9 | 工具系统 + 意图识别 | ⭐⭐ | 一般 |
| Day 10 | 基础设施 + ETL + 前端 | ⭐ | 偶尔 |

---

### 阶段四：补充面试缺口（第 11-13 天）

详见 [第 4 节](#4-面试必备补充知识)。

| 天数 | 主题 | 方式 |
|------|------|------|
| Day 11 | 多 Agent 协作 | 看 LangGraph 文档 + CrewAI 示例 |
| Day 12 | MCP 协议 | 读 Anthropic 官方 MCP 规范 |
| Day 13 | 高级 RAG + 可观测性 | 了解 GraphRAG、LangFuse |

---

### 阶段五：面试模拟（第 14 天）

详见 [第 5 节](#5-面试话术模板)。

- 练习"介绍项目"的 3 分钟版本和 8 分钟版本
- 准备 10 个高频追问的回答
- 主动暴露项目不足 + 说出改进方案

---

## 3. 模块详解

### 3.1 Agent 核心循环

#### 3.1.1 ReAct Agent (`react_agent.py`)

**文件位置**：[app/core/agent/react_agent.py](../app/core/agent/react_agent.py)（291 行）

**一句话**：让 LLM 边思考边行动，像人类解决问题一样。

**核心算法 — ReAct 循环**：

```
用户问题: "今天北京天气怎么样？"
    │
    ▼
Step 1: Thought: 需要查天气 → Action: search → Action Input: {"query":"北京天气"}
             → Observation: "北京今天晴，25°C"
    │
    ▼
Step 2: Thought: 已获得天气信息 → Final Answer: "北京今天晴天，气温 25°C"
```

**代码关键点**：

```python
# 1. 正则解析 LLM 输出 — 提取 Thought / Action / Action Input / Final Answer
def _parse_react_step(text: str) -> Dict[str, Any]:
    thought_m = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|\Z)", text, re.S)
    action_m = re.search(r"Action:\s*(\S+)", text)
    input_m = re.search(r"Action Input:\s*(\{[\s\S]*\})", text)
    # ... 返回解析结果

# 2. 主循环 — 最多 max_steps 步
for step_idx in range(self.max_steps):
    raw = await self._llm.acomplete(messages, temperature=0.2)  # 调用 LLM
    parsed = _parse_react_step(raw)                              # 解析输出
    if parsed.get("done"):
        return AgentResult(success=True, final_answer=answer)    # 得到最终答案
    obs = await self._tools.invoke(action, action_input)         # 执行工具
    history_lines.append(f"Observation: {obs}")                  # 追加到历史
# 超过最大步数 → 返回失败
```

**面试怎么讲**：

> "我自研实现了 ReAct 循环，没调 LangChain 封装。核心是正则解析 LLM 输出中的 Thought/Action/Action Input 字段，然后执行工具，把 Observation 追加到上下文中，循环直到得到 Final Answer。我设置了最大 10 步防死循环，解析失败和工具异常都有兜底处理。"

**考查你的面试题**：
- ReAct 和 Chain-of-Thought 有什么区别？
- 为什么 Action Input 要求 JSON 格式？
- max_steps 设多少合适？太大了有什么问题？

---

#### 3.1.2 Plan-and-Execute Agent (`planner.py`)

**文件位置**：[app/core/agent/planner.py](../app/core/agent/planner.py)（330 行）

**一句话**：先把大任务拆成小步骤，再逐个执行，失败了就重新规划。

**核心流程**：

```
用户目标: "帮我分析 Q2 销售数据并写一份报告"
    │
    ▼
[plan 阶段] LLM 生成:
    t1: 查询 Q2 销售数据 (tool: database)
    t2: 计算同比增长率 (tool: calculator)
    t3: 分析趋势与异常 (reasoning)
    t4: 生成报告 (reasoning)
    │
    ▼
[execute 阶段] 逐个执行:
    t1 → 执行成功
    t2 → 执行失败 ❌
    │
    ▼
[replan 阶段] 基于已有结果重规划:
    t2': 用已知数据估算增长率 (reasoning)
    t3': 生成简化报告 (reasoning)
```

**代码关键点**：

```python
async def run_with_replan(self, query, session_id, ...):
    current_plan = await self.plan(query)       # 1. 生成计划
    for attempt in range(self.max_replan_attempts + 1):
        res = await self.execute(current_plan)  # 2. 执行计划
        if res.success:
            return res                          # 3. 成功返回
        current_plan = await self.replan(...)    # 4. 失败 → 重规划
    return AgentResult(success=False, ...)      # 5. 达到重试上限
```

**面试怎么讲**：

> "Plan-and-Execute 分两个阶段：先用 LLM 把用户目标拆成 JSON 格式的子任务列表，每个子任务标 action_type（tool 还是 reasoning），然后顺序执行。如果某个子任务失败，会触发 replan——把已执行结果和错误信息发给 LLM 重新生成计划。重试次数有上限，全失败就由编排器自动降级到 ReAct。"

---

#### 3.1.3 Agent 编排器 (`orchestrator.py`)

**文件位置**：[app/core/agent/orchestrator.py](../app/core/agent/orchestrator.py)（398 行）

**一句话**：总指挥，根据意图选择执行模式，统一管理记忆和追踪。

**核心设计 — 依赖反转 (Dependency Inversion)**：

```python
# 编排器不依赖具体实现，只依赖协议（Protocol）
class ModelRouter(Protocol):
    def get_llm(self, purpose: str) -> Any: ...

class MemoryManager(Protocol):
    async def get_relevant(self, session_id: str, query: str, limit: int) -> List[str]: ...

class Tracer(Protocol):
    def new_trace_id(self) -> str: ...
```

这 5 个 Protocol 意味着：你可以随时替换底层实现（比如把 Redis 换成 MongoDB），编排器代码不需要改一行。

**执行流程**：

```
run(user_input, session_id, mode)
    │
    ├─ mode == "plan_execute" ?
    │   ├─ Yes → _run_plan_execute()
    │   │          └─ 失败? → degraded=True → _run_react()  # 自动降级
    │   └─ No  → _run_react()
    │
    ├─ enable_reflection?
    │   └─ Yes → _run_reflection()  # 独立 LLM 质检
    │
    └─ 返回 AgentResponse (含 answer + trace_id + steps + reflection)
```

**面试怎么讲**：

> "编排器用了策略模式和依赖反转。通过 Protocol 定义接口，LLM、记忆、工具、追踪都是可替换的。核心逻辑是：根据意图选 ReAct 或 Plan-Execute，Plan 失败自动降级到 ReAct 保证可用性，最后跑一次反思质检。全流程都有 trace 记录。"

---

### 3.2 RAG 全链路

**文件位置**：[app/core/rag_service.py](../app/core/rag_service.py)（689 行，项目最大文件）

**一句话**：让 LLM 能"查阅"你的私有文档再回答问题。

#### 整体流程

```
文档上传
    │
    ▼
[ETL] 解析(PDF/Word/TXT) → 分块(512字/50字重叠) → 向量化(384维)
    │
    ▼
[存储] Milvus Lite (向量) + BM25 内存索引 (关键词)
    │
    ▼
用户提问
    │
    ▼
[检索] 三种模式：
    ├─ vector:  语义向量检索 (COSINE 相似度)
    ├─ keyword: BM25 关键词检索 (IDF + TF)
    └─ hybrid:  RRF 融合 = 1/(k+rank_vector+1) + 1/(k+rank_keyword+1)
    │
    ▼
[生成] 把检索到的文档片段 + 用户问题 → 发给 LLM → 带引用标注的回答
```

#### 3.2.1 BM25 自研实现

这是项目的亮点之一，**从零实现而非调库**。

```python
class _BM25:
    """内存 BM25 (Okapi) 关键词索引"""
    def __init__(self, k1=1.5, b=0.75):  # k1: 词频饱和度, b: 文档长度归一化

    def add(self, doc_id, text):
        # 1. 分词（中英文连续字符）
        # 2. 统计词频 (TF)
        # 3. 更新逆文档频率 (DF) 和 IDF
        # IDF = log(1 + (N - df + 0.5) / (df + 0.5))

    def search(self, query, top_k):
        # BM25 公式：
        # score = Σ IDF(t) * (f * (k1+1)) / (f + k1*(1 - b + b*dl/avgdl))
```

**面试怎么讲**：

> "BM25 是我从零实现的 Okapi 算法。核心是 IDF 计算词的重要性、TF 做词频饱和度处理（k1=1.5）、文档长度归一化（b=0.75 控制长度惩罚）。和直接用 elasticsearch 不同，我理解了每一步的数学原理，面试可以手写公式。"

#### 3.2.2 RRF 混合检索融合

```python
def hybrid_search(self, query_vec, query_text, top_k):
    vec_results = self.vector_search(query_vec, top_k * 2)  # 语义召回
    kw_results = self.keyword_search(query_text, top_k * 2)  # 关键词召回

    rrf_k = 60  # RRF 常数
    for rank, r in enumerate(vec_results):
        scores[r.id] = 1.0 / (rrf_k + rank + 1)   # 排名越靠前分越高
    for rank, r in enumerate(kw_results):
        scores[r.id] += 1.0 / (rrf_k + rank + 1)  # 两边都命中的累加

    return sorted by merged_score[:top_k]
```

**面试怎么讲**：

> "混合检索用 RRF（Reciprocal Rank Fusion）融合向量和关键词的结果。核心思想是不关心绝对分数，只关心相对排名——排名靠前的在两边都会得到高 RRF 分。k=60 是经验值，目的是防止单个第一名主导结果。"

#### 3.2.3 优雅降级设计

```python
# 四层降级链
try:     embedding = sentence-transformers (384维语义向量)  # 最佳
except:  embedding = 字符n-gram哈希 (256维)                  # 降级1

try:     store = Milvus Lite (COSINE向量检索)                # 最佳
except:  store = 内存list + 暴力搜索                         # 降级2
```

零依赖也能跑，上了 Milvus 就更快。这个设计思想在面试中很加分。

---

### 3.3 记忆系统

**核心文件**：
- [app/core/memory/short_term.py](../app/core/memory/short_term.py)（206 行）— Redis 短期记忆
- [app/core/persistent_memory.py](../app/core/persistent_memory.py)（298 行）— 三级记忆管理

#### 三级存储架构

```
写入: append_turn(session_id, role, content)
    ├─ 1. 内存 dict（始终写入，最终兜底）
    ├─ 2. PostgreSQL（主路径，持久化）
    └─ 3. Redis（热路径，加速读取）

读取: get_relevant(session_id, query)
    ├─ 1. Redis → 命中，快速返回  ← 热路径
    ├─ 2. PostgreSQL → Redis 挂了用这个
    └─ 3. 内存 dict → PG 也挂了用这个
```

#### 滑动窗口 + 摘要压缩

```python
class ShortTermMemory:
    window_size = 20    # 最多保留 20 条消息
    max_tokens = 4000   # 超过 4000 token 触发压缩

    async def _compress_if_needed(self, session_id):
        if n <= 20 and total_tokens <= 4000:
            return  # 没超，不用压缩

        # 保留最近 10 条，把更早的发给 LLM 做摘要
        keep = 10
        to_summarize = msgs[:-keep]   # 需要压缩的旧消息
        tail = msgs[-keep:]           # 保留的最近消息

        summary = await llm.ainvoke("请将这些对话压缩为摘要: ...")

        # 替换 Redis 中的内容: [摘要] + [最近10条]
        await redis.delete(key)
        await redis.rpush(key, summary_msg, *tail)
```

**面试怎么讲**：

> "记忆系统用了三级存储：Redis 做短期的滑动窗口加自动摘要压缩，PostgreSQL 做持久化，内存 dict 做最终降级。读取优先级是 Redis → PG → 内存，写入双向同步。压缩策略是保留最近 10 条消息，更早的用 LLM 压缩成摘要。tiktoken 做精确 token 计数。"

---

### 3.4 工具系统

**核心文件**：
- [app/core/tools/base.py](../app/core/tools/base.py) — 工具基类
- [app/core/tools/registry.py](../app/core/tools/registry.py) — 注册表
- [app/core/tools/builtin/](../app/core/tools/builtin/) — 内置工具

```
注册表模式:
  ToolRegistry
    ├── calculator   →  安全数学表达式求值（AST 白名单，不能执行任意代码）
    ├── search       →  DuckDuckGo / SerpAPI 网络搜索
    └── database     →  只读 SQL 查询（检查危险关键词防止注入）

BaseTool 接口:
  - name: str              # 工具名称，Agent 用它来选择
  - description: str       # 工具描述，写入 prompt
  - parameters: List[ToolParameter]  # 参数定义
  - execute(**kwargs)      # 执行工具
  - schema_parameters()    # 导出 OpenAI 格式的参数定义
```

**面试怎么讲**：

> "工具系统基于注册表模式。每个工具继承 BaseTool，定义 name、description、parameters。Agent 通过 prompt 中的工具描述来决定调用哪个。安全方面：计算器用 AST 白名单限制操作符，数据库查询只允许 SELECT 且过滤危险关键词。"

---

### 3.5 模型路由与韧性

#### 3.5.1 三态熔断器 (`circuit_breaker.py`)

**文件位置**：[app/infrastructure/llm/circuit_breaker.py](../app/infrastructure/llm/circuit_breaker.py)（128 行）

**一句话**：当 LLM 服务挂了，自动切断请求，过一阵再试探。

```
          ┌──────────┐
          │  CLOSED  │  正常放行，累计失败次数
          └────┬─────┘
  失败 ≥ 5 次  │  每次成功重置计数
          ┌────▼─────┐
          │   OPEN   │  立即拒绝（快速失败），等待 60s
          └────┬─────┘
     60s 后    │
          ┌────▼──────┐
          │ HALF_OPEN │  允许最多 3 次试探
          └────┬──────┘
    试探成功   │   试探失败
          ┌────▼─────┐ ┌──────────┐
          │  CLOSED  │ │   OPEN   │
          └──────────┘ └──────────┘
```

**代码关键点**：

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60.0, half_open_max=3):

    async def call(self, func, *args, **kwargs):
        # OPEN 状态 → 检查恢复超时
        if self._state == CircuitState.OPEN:
            if 距上次失败 ≥ 60s:
                self._state = CircuitState.HALF_OPEN  # 进入试探
            else:
                raise RuntimeError("熔断器拒绝调用")

        # HALF_OPEN 状态 → 限制试探次数
        if self._state == CircuitState.HALF_OPEN:
            if 试探次数 ≥ 3:
                raise RuntimeError("试探次数已达上限")

        # 执行调用
        try:
            result = await func(*args, **kwargs)
            self._record_success()   # 成功 → 回到 CLOSED
            return result
        except:
            self._record_failure()   # 失败 → 计数+1
            raise
```

**面试怎么讲**：

> "这个三态熔断器复刻了 Nygard 的经典设计模式。Closed 状态计数失败，到 5 次跳闸到 Open 快速拒绝，60 秒后进入 Half-Open 允许最多 3 次试探。用 asyncio.Lock 保证状态切换的并发安全，`time.monotonic()` 不受系统时间调整影响。"

#### 3.5.2 多模型路由 (`model_router.py`)

**文件位置**：[app/infrastructure/llm/model_router.py](../app/infrastructure/llm/model_router.py)（172 行）

**核心算法 — 同优先级加权随机**：

```python
def _select_candidates(self, model_preference):
    # 同优先级内按权重随机：
    # score = random.random() ** (1.0 / weight)
    # weight 越大，越容易被排到前面
    scored = [(cfg, random.random() ** (1.0 / max(cfg.weight, 0.01))) for cfg in group]
    scored.sort(key=lambda x: -x[1])
```

**面试怎么讲**：

> "模型路由支持多模型优先级调度。同一优先级的模型按权重随机分配流量（权重大的被选中的概率高）。每个模型有独立熔断器，调用失败自动切换到下一个候选模型。这比简单轮询更灵活，而且熔断保护避免了雪崩。"

---

### 3.6 意图识别

**文件位置**：[app/core/intent/recognizer.py](../app/core/intent/recognizer.py)

**一句话**：根据用户输入的关键词，判断他要做什么（问答/操作/上传文档）。

**两层意图树**：

```
用户输入: "帮我查一下React和Vue的区别"
    │
    ▼
关键词匹配:
  question → knowledge_qa    ← 命中"区别"关键词
  置信度 = 0.45 + 0.12 × 命中数 = 0.57
```

```
意图树结构:
├── question
│   ├── knowledge_qa     ← "是什么" "怎么做" "区别" "原理"
│   └── comparison       ← "对比" "哪个好"
├── task
│   ├── search           ← "搜索" "查找"
│   ├── calculation      ← "计算" "算一下"
│   └── query            ← "查询"
└── document
    ├── upload            ← "上传" "导入"
    └── download          ← "下载" "导出"
```

**⚠️ 注意**：这是项目最薄弱的部分。关键词硬匹配在实际生产中完全不够用。

**面试怎么讲**：

> "当前版本用规则匹配做意图分类，这是为了快速跑通流程。生产环境应该换成 LLM 分类或微调 BERT 做意图识别。我知道怎么做——把意图标签 + 样本喂给 LLM 做 few-shot 分类，或者用 LangFuse 标注数据后微调。"

---

### 3.7 反思质检

**文件位置**：[app/core/agent/reflection.py](../app/core/agent/reflection.py)

**一句话**：用一个独立的 LLM 调用给 Agent 的回答打分、检查幻觉。

```python
class ReflectionReport:
    quality_score: float       # 0-100 综合评分
    likely_hallucination: bool # 疑似幻觉？
    completeness: str          # "完整"/"不完整"/"部分完整"
    improvement_suggestions: List[str]  # 改进建议
```

**⚠️ 当前局限**：
- 只做评审，不把结果反馈给 Agent 做重新生成（即不是真正的 Reflexion 模式）
- `evidence_snippets` 参数定义了但没实际传入，幻觉检测完全靠 LLM 自己判断

**面试怎么讲**：

> "当前反思是后置的独立评审。如果要做到真正的 Reflexion——把反思结果反馈给 Agent 重新生成——只需要在 orchestrator 里加一个循环：如果质量分低于阈值且幻觉标志为 true，就把改进建议作为额外上下文再跑一次 Agent。这个改动我清楚怎么做。"

---

### 3.8 ETL 文档处理

**核心文件**：
- [app/etl/parser.py](../app/etl/parser.py) — PDF/Word/TXT 解析
- [app/etl/chunker.py](../app/etl/chunker.py) — 文本分块
- [app/etl/pipeline.py](../app/etl/pipeline.py) — ETL 流程编排

```
文件上传 → Parser(根据 MIME 类型选解析器) → Chunker(512字/64字重叠) → 向量化 → 入库
```

**面试知识点**：
- **分块策略**：chunk_size=512 字符、chunk_overlap=64（12.5% 重叠），平衡上下文完整性和检索精度
- **为什么不按句子分块？** 代码里用字符数分块更简单，但丢失了语义边界。生产环境可以用 `LangChain RecursiveCharacterTextSplitter`

---

### 3.9 基础设施

| 组件 | 文件 | 面试知识点 |
|------|------|-----------|
| PostgreSQL | [database/models.py](../app/infrastructure/database/models.py) | SQLAlchemy 2.0 异步 ORM，5 张表 |
| Redis | [cache/redis_cache.py](../app/infrastructure/cache/redis_cache.py) | AOF 持久化，redis-py async |
| Milvus | [vectordb/milvus_client.py](../app/infrastructure/vectordb/milvus_client.py) | Milvus Lite 本地模式，COSINE 相似度 |
| 追踪 | [trace/tracer.py](../app/infrastructure/trace/tracer.py) | 内存 trace（生产需换 OpenTelemetry） |
| LLM | [llm/](../app/infrastructure/llm/) | OpenAI 兼容 API，任何兼容端点都能用 |

---

### 3.10 前端可视化

| 组件 | 作用 |
|------|------|
| `AgentSteps.tsx` | 实时展示 Thought → Action → Observation 步骤 |
| `IntentBadge.tsx` | 意图标签 + 置信度颜色编码 |
| `ToolCallLog.tsx` | 工具调用记录 + 反思分数 |
| `RAGContext.tsx` | 展示检索到的文档片段 + 引用标注 |
| `SearchCompare.tsx` | 向量/关键词/混合三种模式并行对比 |
| `SessionList.tsx` | 会话 CRUD |

---

## 4. 面试必备补充知识

### 4.1 多 Agent 协作（⭐⭐⭐ 必问）

**现状**：项目是单 Agent 架构。

**你需要知道**：

| 模式 | 原理 | 框架 |
|------|------|------|
| Boss-Worker | 一个 Agent 分解任务，多个 Worker 并行执行 | AutoGen, CrewAI |
| Pipeline | Agent A 输出 → Agent B 输入，链式处理 | LangGraph |
| Joint Discussion | 多个 Agent 讨论后投票，减少偏见 | AutoGen GroupChat |
| Hierarchical | 多层嵌套，上级拆解 → 下级执行 → 汇总 | LangGraph StateGraph |

**面试话术**：

> "项目目前是单 Agent，但我知道多 Agent 怎么做。LangGraph 的 StateGraph 可以定义多个 Agent 节点，通过条件边实现 Boss-Worker 模式。核心难点是通信协议和状态共享——AutoGen 用消息总线，CrewAI 用共享上下文。"

**动手了解**：跑一遍 [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)。

---

### 4.2 MCP 协议（⭐⭐⭐ 必问）

**现状**：传统 Function Calling（代码里硬编码工具函数）。

**MCP vs Function Calling 对比**：

| | Function Calling | MCP |
|------|------|------|
| 工具定义 | 代码内 | 独立 MCP Server |
| 通信方式 | 直接函数调用 | JSON-RPC 2.0 |
| 工具发现 | 启动时注册 | 运行时动态发现 |
| 热插拔 | ❌ | ✅ |
| 跨语言 | ❌ | ✅ 任何语言 |
| 安全隔离 | ❌ | Server 进程级隔离 |

**面试话术**：

> "当前项目用 Function Calling，工具函数在代码里硬编码。MCP 的核心价值是工具与 Agent 解耦——工具作为独立 Server 进程运行，Agent 通过 JSON-RPC 动态发现和调用。支持热插拔和跨语言，Anthropic 官方已经推了。我知道升级路径：把现有工具封装成 MCP Server，Agent 侧加一个 MCP Client 适配层。"

---

### 4.3 高级 RAG 模式（⭐⭐）

| 模式 | 一句话 | 解决什么问题 |
|------|--------|-------------|
| Agentic RAG | Agent 自己决定要不要检索 | 不该检索时不浪费 |
| Self-RAG | 检索后先自我评审质量 | 检索结果差时重试 |
| Corrective RAG | 检索不好时自动改写查询 | 查询词不精准 |
| GraphRAG | 知识图谱 + RAG | 实体关系推理 |
| Multi-hop RAG | 多次检索，逐跳深化 | 需要多步推理的复杂问题 |

---

### 4.4 生产可观测性（⭐⭐）

**现状**：`InMemoryTracer` — 只在内存里记录，进程重启就没了。

**你需要知道**：

| 工具 | 用途 |
|------|------|
| **LangFuse** | LLM 专用：prompt 版本管理、token 统计、trace 可视化 |
| **OpenTelemetry** | 通用分布式追踪标准 |
| **Prometheus + Grafana** | 指标收集 + 仪表盘 |
| **ELK / Loki** | 结构化日志收集与检索 |

---

### 4.5 安全与护栏（⭐）

你需要知道的防御措施：

| 风险 | 方案 |
|------|------|
| Prompt Injection | 输入清洗 + 角色分离 + 指令优先级 |
| 工具滥用 | 权限分级 + 参数白名单 + 操作审计 |
| 数据泄露 | PII 脱敏 + 输出过滤 |
| 幻觉 | Reflection 质检 + 引用溯源 |

---

## 5. 面试话术模板

### 5.1 "介绍一下你最熟悉的项目"（3 分钟版）

> 我做了一个企业级 AI Agent 智能客服平台，技术栈是 Python FastAPI + React。
>
> **核心架构**是自研的 ReAct 和 Plan-and-Execute 双模式 Agent。ReAct 是 Thought → Action → Observation 循环，我没调 LangChain 封装，而是用正则解析 + JSON 结构化输出自己实现的。Plan-and-Execute 是 LLM 先拆子任务再逐个执行，失败会触发 replan 重规划，再失败就自动降级到 ReAct。
>
> **RAG 链路**我从零实现了 BM25 关键词检索（Okapi 算法），配合 Milvus 向量检索做 RRF 混合融合，支持向量/关键词/混合三种模式。
>
> **记忆系统**用了三级存储：Redis 做短期滑动窗口加自动摘要压缩，PostgreSQL 做持久化，内存 dict 做最终兜底，任何一层挂了都能自动降级。
>
> **工程韧性**上实现了三态熔断器 + 多模型优先级路由 + 自动降级，保证 LLM 服务不可用时系统仍然有响应。
>
> 前端做了 Agent 步骤实时可视化、搜索模式对比面板、意图标签展示。Docker Compose 一键部署。

### 5.2 "这个项目还缺什么"（主动暴露 + 改进方案）

> 如果继续做，我会优先补三个方向：
>
> **1. 多 Agent 协作**：目前是单 Agent。我会用 LangGraph 的 StateGraph 引入 Boss-Worker 模式，让 Worker Agent 并行执行子任务。核心难点是状态共享和通信协议。
>
> **2. MCP 协议升级**：当前是硬编码 Function Calling。我会把工具封装成独立 MCP Server，Agent 通过 JSON-RPC 动态发现。好处是支持热插拔、跨语言、进程级安全隔离。
>
> **3. 生产可观测性**：当前用内存 trace，没有持久化。我会接入 LangFuse 做 prompt 调试和 trace 可视化，Prometheus + Grafana 做指标监控和告警。

### 5.3 高频追问速答

| 问题 | 回答要点 |
|------|---------|
| **ReAct 死循环怎么办？** | max_steps=10 硬限制；每步都有 try/except；工具调用超时；编排器层面还有整体异常兜底 |
| **RAG 检索不准怎么办？** | 三种模式可切换；RRF 融合取交集；chunk_overlap 保证边界不丢信息；生产环境可加 Cross-Encoder 重排序 |
| **LLM 挂了怎么办？** | 三态熔断器快速拒绝 → 多模型路由自动切换备选 → 返回降级响应而非白屏 |
| **记忆无限增长怎么办？** | 滑动窗口限制条数 + tiktoken 计数触发 LLM 摘要压缩 + 只保留最近 10 条原文 |
| **BM25 和向量检索各有什么优劣？** | BM25 对精确关键词匹配好（如人名、型号）、不依赖 GPU；向量检索对语义相近的也能召回。两者互补 |
| **为什么用 RRF 而不是简单加权？** | 向量和 BM25 的分数尺度不同，不能直接相加。RRF 只看排名、不关心绝对值，天然适合异质检索结果融合 |

---

## 附录：关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| ReAct | Reasoning + Acting：边想边做，交替进行 |
| Plan-Execute | 先规划再执行：拆子任务 → 逐个做 → 失败重规划 |
| RRF | 倒数排名融合：1/(k+rank+1)，只关心排名不关心绝对分数 |
| BM25 | 经典关键词检索算法：IDF × TF饱和度 × 文档长度归一化 |
| 三态熔断 | Closed→Open→Half-Open，防止雪崩 |
| 依赖反转 | 依赖抽象接口而非具体实现 (Protocol) |
| 滑动窗口 | 只保留最近 N 条记录，旧的删除/压缩 |
| SSE | Server-Sent Events：服务端单向推送流式数据 |
| COSINE | 余弦相似度：衡量两个向量方向的接近程度 |
| MCP | Model Context Protocol：工具与 Agent 解耦的标准协议 |

---

*最后更新：2026-07-20*
