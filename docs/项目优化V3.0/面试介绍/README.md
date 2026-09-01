# 心血管慢病管理 AI Agent 平台 — 面试讲解稿

## 项目来源（不主动提，被问时如实说）

> 本项目基于 GitHub 开源项目 [ai-agent-interview-guide](https://github.com/bcefghj/ai-agent-interview-guide) 做了大量二次开发和功能扩展，包括：反思闭环、四层记忆读写打通与日志、Evidence Collector、SelfCorrector、领域安全策略、前端反思结果展示等。原始项目提供了基础框架（FastAPI + React + ReAct + RAG），我在上面理解每行代码后，做了深度定制。

---

## 一句话概述

> 一个面向心血管慢病管理场景的 AI Agent 对话平台，覆盖"意图理解 → 模式路由 → ReAct 执行 → 反思闭环 → 四层记忆"完整链路。后端 FastAPI + 前端 React，LangChain 用于 Function Calling 工具调用。

---

## 一、项目速览

| 维度 | 说明 |
|------|------|
| **业务** | 心血管慢病管理与健康咨询 |
| **用户** | 患者/家属（对话）+ 医生（文档上传） |
| **规模** | ~3000 行核心 Python + ~600 行前端 TSX，33 个依赖 |
| **部署** | docker-compose 一键启动（Redis + PG + 应用） |
| **测试** | 反思闭环 17 个单元测试，pytest-asyncio |

---

## 二、整体架构（面试时可直接画这张图）

```
前端 React (ChatPage)
  │ POST /api/v1/chat/agent
  ▼
AgentService.chat()
  │
  ├─ ① IntentRecognizer ──→ 意图 (LLM分类 / 关键词降级)
  ├─ ② ModeRouter      ──→ 模式 (规则快路径 / LLM Router)
  │
  ▼
AgentOrchestrator.run()
  │
  ├─ ③ ReAct 执行 (FC优先 → 兼容降级)
  │     FunctionCallAgent (LangChain bind_tools)
  │     ReActAgent (正则解析版，兼容后备)
  │     PlannerAgent (Plan→Execute→Replan)
  │
  └─ ④ 反思闭环
       Evidence收集 → 评分 → ≥80过 / 60-79修正 / <60重试
  │
  ▼
记忆写入 (四层并行, 互不阻塞)
  L1 内存 → L3 PG → L2 Redis → L4 Milvus
```

---

## 三、六个技术模块（按讲解优先级排列）

---

### 3.1 反思闭环（最有区分度，重点准备）

**做了什么**：Agent 生成回答后，加了一层带证据的质量控制，不是直接返回。

**三步流程**：

```
① 收集证据 → 从执行步骤中提取 rag_search/web_search 的实际返回内容
② 反思评估 → LLM 对照 用户问题+回答+证据，打出 quality_score，判断是否幻觉
③ 三级响应 →
    ≥80分 → 直接返回
    60-79 → SelfCorrector 单次 LLM 修复（比重跑 Agent 省 token）
    <60  → 携带反思反馈，重新执行 Agent（最多 2 次）
```

**为什么加 Evidence Collector**：反思不能靠 LLM 凭空判断幻觉。Agent 调了 `rag_search("高血压指南")` 返回了 A/B/C，但回答说了 D——反思 LLM 对照证据后就能发现这是编的。

**成本控制**：高质量回答只多 1 次反思调用。只有低质量才会触发重试，而这恰恰是最值得额外投入的场景。SelfCorrector 的定位是：修复措辞/删幻觉片段/补遗漏，1 次 LLM 调用 vs 完整 N 步 ReAct 循环，省了很多 token。

> **讲解话术**："这个项目我花最多时间的是反思闭环。基本思路是：Agent 生成回答后不直接返回，先收集它调用工具时拿到的实际内容作为证据，再让 LLM 对照证据评审回答质量。80 分以上直接过，60 到 79 用 SelfCorrector 轻量修一下，60 分以下触发 Agent 重试并把上一轮的问题反馈带进去。做完之后写了 17 个测试，覆盖三等分数和各种边界情况。"

**代码**：[reflection.py](../../app/core/agent/reflection.py)、[evidence_collector.py](../../app/core/agent/evidence_collector.py)、[orchestrator.py:306-367](../../app/core/agent/orchestrator.py#L306-L367)

---

### 3.2 Agent 执行引擎

**两个实现，一条链路**：

| | FunctionCallAgent | ReActAgent |
|---|---|---|
| **状态** | 优先路径 | 降级后备 |
| **工具调用** | LangChain `bind_tools()` → 结构化 JSON | 正则解析 `Thought:` / `Action:` 文本 |
| **优势** | 解析失败率 ~0% | 不依赖模型 function calling 能力 |
| **劣势** | 需要模型支持 FC | 解析失败率 ~10-15% |

**实际运行链路**：
```
Orchestrator._invoke_react()
  → 优先走 FunctionCallAgent (结构化 JSON)
  → 失败自动切换到 ReActAgent (正则解析)
```

**关于 ReAct 实现**：`react_agent.py` 从 prompt 构建、Thought/Action 正则解析、for 循环控制步数上限、Observation 注入、工具异常处理，都是逐行理解的代码。框架方面，用 LangChain 的 `ChatOpenAI.bind_tools()` 做 Function Calling——只让它负责"让模型输出结构化 tool_calls"，循环编排逻辑还是自己控制。

> **讲解话术**："Agent 执行这块，项目里有两个版本在跑——优先走 Function Calling，用 LangChain 的 bind_tools 注册工具 Schema，模型直接返回结构化 JSON。如果模型不支持 FC 或者调用失败，自动降级到正则解析版——从模型输出里用正则提取 Thought 和 Action。两个版本共享同一套编排逻辑。"

**代码**：[react_agent.py](../../app/core/agent/react_agent.py)、[function_call_agent.py](../../app/core/agent/function_call_agent.py)

---

### 3.3 四层记忆 + 逐层降级

**每轮对话 2 读 2 写**：

```
读取（优先级链）: Redis 热路径 → PG 冷存储 → 内存兜底
                  每层都叠加 Milvus 语义召回 (top_k=3)

写入（并行旁路）: 内存(同步) | PG(主持久化) | Redis(滑动窗口) | Milvus(旁路)
                  各层 try/except 隔离，失败不阻塞其他层
```

**各层定位**：

| 层 | 存储 | 角色 | 生命周期 |
|------|------|------|------|
| L1 | 内存 dict | 最终兜底，始终可用 | 进程重启丢失 |
| L2 | Redis List | 热路径加速，滑动窗口 + LLM 摘要压缩 | 可配置 TTL |
| L3 | PostgreSQL | 持久化真相源 | 永久 |
| L4 | Milvus | 语义向量召回（历史相似记忆） | 永久 |

**已知问题（主动提）**：`_pg_ok` 和 `_redis_ok` 是单向熔断——PG 故障一次后就不再走 PG，直到重启。正确的做法是三态断路器（Closed→Open→Half-Open），PG 恢复后自动回连。这个还没改，写在改进计划里了。

> **讲解话术**："记忆部分做了四层——内存兜底、Redis 做热缓存、PG 做持久化、Milvus 做语义召回。读走优先级链降级，写走并行旁路。我在文档里写清楚了每轮对话各层调了几次、读写了什么字段、降级路径是什么。也诚实分析了当前的问题——单向熔断，PG 挂一次后就不再用了，应该改成三态断路器。"

**代码**：[persistent_memory.py](../../app/core/persistent_memory.py)

---

### 3.4 意图识别 + 模式路由

```
用户输入
  ├─ LLM 分类 → 5 类意图
  │   注入对话历史，处理"那它呢？"这类指代
  │
  └─ 关键词降级 → 规则树匹配（LLM 不可用时的兜底）

ModeRouter
  ├─ 规则快路径 → 高置信度 + 明确映射 → 不调 LLM
  ├─ LLM Router → 低置信度 / 模糊场景 → 二次判断
  └─ react 兜底
```

> **讲解话术**："意图识别做了双通道——LLM 分类为主，关键词规则为降级。模式路由也是两级，规则快路径处理大部分场景不用额外调 LLM，模糊的才走 LLM Router。"

**代码**：[intent/recognizer.py](../../app/core/intent/recognizer.py)、[agent/mode_router.py](../../app/core/agent/mode_router.py)

---

### 3.5 领域安全边界

```python
DomainPolicy:  # 数据类定义领域安全策略
  risk_keywords     → "胸痛""呼吸困难" → 立即返回 120 分流，不进入 Agent
  emergency_notice  → 急诊提示文本
  disclaimer        → 回答末尾追加医疗免责
```

> **讲解话术**："医疗场景安全很重要。我做了一个 DomainPolicy 数据类——检测到急诊关键词直接返回 120 分流提示，短路后面的 Agent 工具调用。当前有心脑血管和通用两个策略，加新领域只需加一个配置实例。"

**代码**：[domain/policy.py](../../app/core/domain/policy.py)

---

### 3.6 RAG 混合检索

| 模式 | 实现 |
|------|------|
| 向量 | Milvus Lite + all-MiniLM-L6-v2 (384-dim COSINE) |
| 关键词 | BM25 (Okapi) + jieba 中文分词 |
| 混合 | RRF 倒数排名融合 (k=60) |
| 降级 | Milvus 不可用 → 字符 n-gram 嵌入 + 内存暴力搜索 |

> **讲解话术**："RAG 部分是向量+关键词混合检索，RRF 融合。做了完整的降级链——Milvus 不可用时自动切换到纯内存实现，零外部依赖也能跑。"

**代码**：[rag_service.py](../../app/core/rag_service.py)

---

## 四、面试自述模板

### Part 1：项目介绍（45 秒）

> "我做的是一个心血管慢病管理的 AI Agent 对话平台。后端 FastAPI、前端 React。主要链路是：用户发消息 → 意图识别 → 选择执行模式 → ReAct Agent 执行工具调用 → 反思评估回答质量 → 四层记忆保存对话历史。这个项目我是基于一个开源项目做了二次开发，在上面加了自己理解和实现的功能。"

### Part 2：展开讲 2 个亮点（各 1.5-2 分钟）

**推荐讲反思闭环 + 四层记忆，这两个准备最充分。**

**反思闭环**：
> "我在这个项目上花最多精力的是反思闭环。基本想法是：Agent 生成回答后，不应该直接给用户，应该先检查一下质量。我做了三步——第一步，把 Agent 执行过程中调工具拿到的实际内容收集起来作为证据；第二步，让反思 LLM 对照证据去评审回答，看看有没有编造的内容、有没有遗漏；第三步，根据评分做不同处理：80 分以上直接返回，60 到 79 用一次 LLM 调用做轻量修复，60 分以下触发 Agent 重新执行，把反思的反馈带进去。这样做的目的是在质量和成本之间取一个平衡。做完后写了 17 个单元测试，覆盖三等分数和各种边界情况。"

**四层记忆**：
> "记忆这块我用四层存储——内存做兜底保证永远可读写，Redis 做热路径加速，PostgreSQL 做全量持久化，Milvus 做语义向量召回。每轮对话都是两读两写，读走优先级链降级，写走并行旁路各层互不阻塞。Redis 那边做了滑动窗口加 LLM 摘要压缩，控制上下文 token 不要爆炸。我在文档里分析了一致性问题——选了 AP 优先于 CP 的路线，因为对话场景对短暂不一致容忍度很高。也指出了当前单向熔断的缺陷和改进方向。"

### Part 3：收尾（30 秒）

> "这个项目让我学到了挺多东西——不是只会调框架 API，而是理解每一步在做什么、为什么这样设计、有什么取舍。每个模块都写了文档，核心逻辑有测试。如果对哪个模块感兴趣我可以展开讲。"

---

## 五、追问速答表

| 问题 | 简短回答 |
|------|------|
| "为什么不用现成框架？" | "用了——LangChain 的 bind_tools 就在生产路径里做 Function Calling。但我也看了它内部怎么做的，所以自己也写了一份正则解析版，理解了 Thought/Action/Observation 循环的细节。" |
| "四层记忆是不是过度设计？" | "每层有各自的定位：内存兜底、Redis 加速、PG 持久化、Milvus 语义召回。每层都可能挂，降级链保证服务不中断。减到两层也成立，但语义召回（Milvus）和热缓存（Redis）在对话场景确实有实际价值。" |
| "反思会不会增加很多成本？" | "高质量的只多 1 次反思调用，可以忽略。只有低质量的才触发重试——这恰恰是最需要投入的场景。SelfCorrector 就是为了省成本设计的。" |
| "数据一致性问题？" | "选了 AP 优先于 CP——对话场景允许短暂不一致。写四份没有分布式事务，各层 try/except 隔离。已知缺陷是单向熔断，改进方向是三态断路器。" |
| "Milvus flush 报错？" | "Milvus Lite 在 Windows 上的已知问题，insert 本身已经成功，flush 失败下次重试就行。代码做了静默处理。" |
| "项目是你自己从头写的吗？" | "基于一个 GitHub 开源项目做了二次开发，我在此基础上理解和学习，然后加了很多功能——反思闭环、Evidence Collector、SelfCorrector、四层记忆日志打通、领域安全策略、前端增强等。每个文件我都读过了，知道为什么这样写。" |
| "你学到了什么？" | "最大的收获是：写代码之前要想清楚设计。反思闭环我画了好几版流程图才开始写。文档里把调用次数、字段、降级路径都算清楚了。还有就是不要回避缺陷——诚实说哪里做得不够好比假装完美更能让人信服。" |

---

## 六、深水区问题（面试官深挖备用）

**Q: Evidence Collector 的证据本身可能不完整，怎么保证判断准确？**

> 它不保证证据完整——它保证的是反思 LLM 看到的东西是 Agent 实际获取到的。比如 Agent 搜"高血压指南"系统返回了三条内容，但它回答声称系统提到了第四条——反思 LLM 对照就能发现这是编的。关键是"对照"，不是"穷举"。证据缺失的风险远小于让 LLM 凭空的判断。

**Q: SelfCorrector 和完整重试的边界为什么在 60 分？**

> 60 分以下意味着方向可能错了——可能搜错了关键词、理解偏了——需要重新调工具获取正确信息，SelfCorrector 做不到，因为它只是单次 LLM 调用，不能调工具。60-79 是大方向对但细节有问题，适合轻量修复。60 这个阈值目前是个经验值，实际需要 A/B 测试校准。

**Q: PG 和 Redis 同时故障怎么办？**

> 内存兜底，当前对话不中断。但重启后丢失——这是单向熔断最脆弱的地方。改进方向：降级期间写 WAL 日志，PG 恢复后回放。核心是把熔断器改成三态，PG 恢复了能自动重新启用。

---

## 七、文件结构

```
ai-agent-interview-guide/
├── app/
│   ├── core/
│   │   ├── agent/
│   │   │   ├── orchestrator.py      # 编排器：ReAct/Plan-Execute + 反思闭环集成
│   │   │   ├── react_agent.py       # ReAct 正则解析版（FC 降级后备）
│   │   │   ├── function_call_agent.py # FC 版（LangChain bind_tools，优先路径）
│   │   │   ├── planner.py           # Plan-and-Execute 规划器 + 重规划
│   │   │   ├── mode_router.py       # 意图驱动的模式路由器
│   │   │   ├── reflection.py        # ⭐ 反思 + SelfCorrector + 重试上下文
│   │   │   └── evidence_collector.py # 从 steps 提取工具调用证据
│   │   ├── persistent_memory.py     # ⭐ 四层记忆统一管理器
│   │   ├── rag_service.py           # ⭐ RAG: Milvus + BM25 + RRF
│   │   ├── intent/recognizer.py     # 意图识别：LLM + 关键词双通道
│   │   ├── domain/policy.py         # 领域安全策略（可插拔）
│   │   ├── memory/short_term.py     # Redis 滑动窗口 + LLM 压缩
│   │   └── memory/long_term.py      # Milvus 长期语义记忆
│   └── models/schemas.py            # Pydantic 数据模型
├── frontend/src/
│   ├── pages/ChatPage.tsx           # 主页面：三种模式 + 自动路由
│   └── components/
│       ├── ToolCallLog.tsx           # 反思结果：幻觉警告/修正标记/评分
│       └── AgentSteps.tsx            # ReAct 步骤可视化
├── tests/
│   └── test_reflection_loop.py      # 反思闭环 17 个测试
└── docs/项目优化V3.0/
    ├── 记忆层.md                     # 四层记忆调用全貌 + 一致性分析
    └── 反思评估机制设计.md            # 反思闭环设计文档
```

---

## 八、技术栈

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
| 测试 | pytest + pytest-asyncio | 异步 UT |

---

## 九、已知不足与改进方向

| 优先级 | 问题 | 方案 |
|------|------|------|
| P0 | 记忆层单向熔断 | `_pg_ok` 改为三态断路器，PG 恢复后自动回连 |
| P1 | 降级期数据可能丢失 | PG 不可用时写 WAL，恢复后回放 |
| P1 | 无流式输出 | SSE 词级推送 |
| P2 | 工具结果无缓存 | rag_search 短期缓存同 query |
| P2 | 评测体系缺失 | 自动评测：正确性/完整性/安全性 |

> 面试官想看到的不是"完美架构"，而是"知道哪里还不够好，下一步怎么改"。

---

## 附：LangGraph 实现说明（面试时不提，自己了解即可）

项目中有一份完整的 LangGraph ReAct Agent 实现（[langgraph_agent.py](../../app/core/agent/langgraph_agent.py)，307 行），使用 `StateGraph` + `ToolNode` + `Evaluator` 节点 + 条件边构建，含质量分不达标自动重试逻辑。但这份实现未接入 orchestrator 和 agent_service 的服务链路，不在实际运行路径中。

保留的原因是：理解并对比"框架版"和"手写版"的差异，有助于深入理解 Agent 循环的本质。面试时不必主动提及——如果被问到"用没用 LangGraph"，可以如实说"有一份完整的 StateGraph 实现用于学习对比，但当前运行链路用的是 LangChain 的 bind_tools 加自写的编排逻辑"。
