# Agent 请求处理全链路

> 从用户输入到最终回答，一条请求在系统里经历了什么。按调用顺序逐层展开。

---

## 总览

```
用户输入
  ↓
① 请求校验与领域策略识别（DomainPolicy：高风险关键词 → 安全分流）
  ↓
② 对话历史加载（四层记忆：Milvus 语义 → Redis → PG → 内存）
  ↓
③ 意图识别（LLM 分类 + 关键词降级双通道）
  ↓
④ 模式路由（规则快路径 → LLM Router → react 兜底）
  ↓
⑤ ReAct 循环执行（FC 优先 → 正则降级）
  ↓
⑥ 反思闭环（≥80过 / 60-79修正 / <60重试）
  ↓
⑦ 四层记忆写入 + 回答边界控制
  ↓
返回最终答案 + 执行轨迹
```

---

## ① 请求校验与领域策略识别

**文件**：[domain/policy.py](../../app/core/domain/policy.py)

每个对话入口（`/chat`、`/chat/agent`、`/chat/stream`）的第一行业务逻辑：

```python
policy = get_domain_policy(request.domain)

if policy.is_high_risk(user_text):
    return policy.safety_prefix(user_text)  # 直接返回 120 提示，短路后续所有逻辑
```

### DomainPolicy 数据结构

```python
@dataclass(frozen=True)
class DomainPolicy:
    name: str                        # "general" | "cardiovascular"
    system_prompt: str               # 注入到 LLM system prompt
    risk_keywords: tuple[str, ...]   # 高风险关键词
    emergency_notice: str            # 急诊提示文本
    disclaimer: str                  # 回答末尾免责声明
```

### 三个核心方法

| 方法 | 作用 | 触发时机 |
|------|------|------|
| `is_high_risk(text)` | 正则去空格 → 小写 → 逐关键词匹配 | 每个请求入口 |
| `safety_prefix(text)` | 返回急诊提示（命中关键词时），否则空字符串 | `is_high_risk` 返回 True |
| `with_boundary(answer, query)` | 高风险优先返回急诊提示；普通回答追加免责声明 | 回答返回前 |

### 心血管策略的 12 个急诊关键词

```
胸痛、胸闷伴冷汗、呼吸困难、喘不上气、意识不清、
昏厥、晕厥、口角歪斜、言语不清、一侧肢体无力、
突发剧烈头痛、心跳骤停
```

### 文件上传校验（[upload_validation.py](../../app/api/upload_validation.py)）

```
文件名非空 → 长度≤255 + 无空字符 → 无路径穿越 → 字符白名单
→ 扩展名白名单(.pdf/.txt/.md/.markdown) → 非空 + ≤10MB
```

---

## ② 对话历史加载

**文件**：[persistent_memory.py](../../app/core/persistent_memory.py)

意图识别和模式路由**共享同一份历史**，一次查询，两处复用：

```python
history = await self._memory.get_relevant(session_id, user_text, limit=6)

# 传入意图识别
intent_result = await self._intent_recognizer.recognize(
    user_text, context={"history": history}
)

# 传入模式路由
decision = await self._mode_router.resolve(
    user_text, intent_result, history=history
)
```

### 数据格式

`list[str]`，每个元素 `"{role}: {content}"`：

```python
[
    "[历史记忆] 三个月前讨论过血压监测频率...",   # ← Milvus 语义召回 Top3
    "[历史记忆] 用户之前询问过ARB类药物的适应症...",
    "user: 高血压怎么治疗",
    "assistant: 高血压治疗包括生活方式干预和药物治疗...",
    "user: 那这些药物的副作用呢",
    "assistant: ACEI类药物常见副作用有干咳...",
]
```

### 四层读取优先级

```
① Redis ShortTermMemory（热路径，滑动窗口 + LLM 摘要压缩）
   ↓ 失败 / 不可用
② PostgreSQL SessionRepository（冷存储，全量持久化）
   ↓ 失败 / 不可用
③ 内存 dict（最终兜底，始终可用）

④ 每层出结果后，叠加 Milvus 语义召回（top_k=3）
   语义记忆插入最前面，前缀 "[历史记忆] "
```

### 为什么需要历史

**意图识别**：处理指代消解。没有历史时"那它呢？"被判为闲聊；有历史知道上一轮在问药物副作用，正确判为知识问答。

**模式路由**：LLM Router 做决策时需要知道"用户当前在聊什么"，而非只看当前 query。

### 读取失败处理

三层全部失败 → `history` 为空列表 → 意图识别和模式路由退化为**单轮模式**。

---

## ③ 意图识别

**文件**：[intent/recognizer.py](../../app/core/intent/recognizer.py)

双通道设计：LLM 分类优先，关键词规则降级。

### 五类意图

| 意图 | 触发特征 | 槽位 |
|------|------|------|
| 知识问答 | 什么/为什么/如何/区别 | `compare_entities` |
| 实时搜索 | 今天/最近/最新/2026 | — |
| 数学计算 | 计算/算一下 或 含算式 | `numbers[]` |
| 文档管理 | 上传/下载/删除 | `doc_action` |
| 闲聊对话 | 你好/谢谢/再见 | — |

### LLM 分类（优先）

```
System: 意图分类 prompt（5 类定义 + 示例 + 输出 JSON 格式）
User:   用户输入 + 对话历史
        ↓
LLM (temperature=0.1)
        ↓
正则提取 JSON → IntentResult(intent, confidence, slots)
```

关键设计：
- 注入对话历史处理指代消解
- `_extract_json()` 先 `json.loads`，失败则正则 `\{[\s\S]*\}` 兜底
- LLM 没提取数字槽位时，用正则从 query 补提

### 关键词规则（降级）

维护一个两层规则树，遍历所有关键词计命中次数，取最高分意图：

```
置信度 = 0.45 + 0.12 × 命中次数（上限 0.95）
零命中 → 默认"知识问答"，置信度 0.25~0.35
```

### 降级触发

LLM 分类异常（API key 未配置、网络不通、返回非 JSON 且正则也提取失败）→ `except` 捕获 → 静默走关键词规则。用户无感知。

### 置信度与澄清

普通对话路由（`/chat`）中，置信度 < 0.55 时注入澄清提示，让模型主动追问用户。

---

## ④ 模式路由

**文件**：[mode_router.py](../../app/core/agent/mode_router.py)

三级决策链：规则快路径 → LLM Router → react 兜底。

### ① 规则快路径

意图→模式映射表（零 LLM 调用）：

| 意图 | 默认模式 | 生效门槛 |
|------|------|------|
| 数学计算 | react | conf ≥ 0.70 |
| 实时搜索 | plan_execute | conf ≥ 0.70 |
| 闲聊对话 | react | 无门槛 |
| 知识问答 | react | 无门槛 |
| 文档管理 | react | 无门槛 |

两个拦截条件，满足任一跳过规则：
1. 意图置信度 < 0.50（意图本身不可靠）
2. 意图置信度 < 该意图的生效门槛

**知识问答特殊升级**：query 含时间敏感词（"今天""最新""2026"等）时，从 react 升级为 plan_execute。

### ② LLM Router

规则无法决策时，调 LLM 二次判断：

```
System: 路由规则（react vs plan_execute 的适用场景 + 示例）
User:   用户查询 + 对话历史 + 意图分析结果 + 可用工具列表
        ↓
LLM (temperature=0.1)
        ↓
JSON → 校验 mode 合法性（只认 react / plan_execute）
```

### ③ react 兜底

LLM Router 不可用或调用失败 → `ModeDecision(mode="react", source="fallback")`

### 显式覆盖

前端传了 `agent_mode: "react"` 或 `"plan_execute"` 时，直接使用，跳过 ModeRouter。

---

## ⑤ ReAct 循环执行

**文件**：[orchestrator.py](../../app/core/agent/orchestrator.py)、[function_call_agent.py](../../app/core/agent/function_call_agent.py)、[react_agent.py](../../app/core/agent/react_agent.py)

`_invoke_react()` 优先 Function Calling，失败自动降级正则解析版。

### 两个后端对比

| | FunctionCallAgent | ReActAgent |
|---|---|---|
| 状态 | 优先路径 | 降级后备 |
| 工具注册 | `ChatOpenAI.bind_tools(tools)` 注入 JSON Schema | 文本拼进 prompt 的"可用工具"列表 |
| LLM 返回 | `AIMessage.tool_calls`（结构化 JSON） | 原始文本，正则提取 |
| 解析失败率 | ~0% | ~10-15% |
| 历史传递 | `AIMessage` + `ToolMessage` 追加到消息列表 | 文本拼接 Step N 历史块 |
| 编排控制 | for 循环（自己写） | for 循环（自己写） |

### FunctionCallAgent 循环（优先路径）

```python
for step_idx in range(self.max_steps):           # max_steps=10
    messages = _build_messages(query, system_text, history, mem_snippets)

    # ① 调 LLM（已 bind_tools，模型自动决定是否调工具）
    response: AIMessage = self._llm.invoke(messages)

    if response.tool_calls:                      # ② 模型要求调工具
        history.append(AIMessage(...))            # 追加 assistant 消息
        for tc in response.tool_calls:
            obs = await tool.ainvoke(tc["args"])  # 执行工具
            history.append(ToolMessage(...))      # 追加工具结果
        steps.append({...})
        continue                                  # ← 下一轮循环

    else:                                         # ③ 无 tool_calls → 最终答案
        return AgentResult(success=True, final_answer=response.content)
```

关键：`history` 是 LangChain `BaseMessage` 列表，每轮**累加**，LLM 看到完整的多轮工具调用历史。

### ReActAgent 循环（降级后备）

```python
for step_idx in range(self.max_steps):             # max_steps=10
    user_prompt = build_react_user_prompt(query, tool_desc, history_block)
    raw = await self._llm.acomplete(messages)

    parsed = _parse_react_step(raw)                # 正则提取 Thought/Action/Final Answer

    if parsed.get("done"):                         # 解析到 Final Answer
        return AgentResult(success=True, final_answer=...)

    if not parsed.get("action"):                   # 什么都没解析到 → 失败
        return AgentResult(success=False, error=...)

    obs = await self._tools.invoke(action, args)   # 执行工具
    history_lines.append(f"Step {step_idx+1}\nThought:...\nAction:...\nObservation:...")
    # ← 下一轮，历史通过 user prompt 注入
```

关键区别：ReActAgent 的"历史"是文本拼接，每轮拼进 user prompt 的"已执行的步骤与观察"区域。

### 降级逻辑

```python
if use_function_calling:
    try:
        result = await fc_agent.run(query, ctx)
    except Exception:
        pass  # 走降级
    else:
        if result.success:
            return result  # FC 成功，直接返回

# 降级到 ReActAgent
agent = ReActAgent(llm=..., tools=..., memory=...)
return await agent.run(query, ctx)
```

降级对上层透明——编排器不关心最终是哪个后端完成的。

---

## ⑥ 反思闭环

**文件**：[reflection.py](../../app/core/agent/reflection.py)、[evidence_collector.py](../../app/core/agent/evidence_collector.py)

Agent 生成回答后，加一层带证据的质量控制。

### 三步流程

```
① 收集证据：从执行步骤中提取 rag_search/web_search 的实际返回内容
        ↓
② 反思评估：LLM 对照 用户问题+回答+证据，打出 quality_score，判断是否幻觉
        ↓
③ 三级响应：
   ≥80 分 → 直接返回（只多 1 次反思调用）
   60-79 分 → SelfCorrector 单次 LLM 修复（比重跑 Agent 省 token）
   <60 分 → 携带反思反馈，重新执行 Agent（最多 2 次）
```

### 为什么加 Evidence Collector

Agent 调了 `rag_search("高血压指南")` 返回了 A/B/C，但回答说了 D——反思 LLM 对照证据后就能发现这是编的。关键是"对照"，不是"凭空判断"。

### 重试逻辑

```python
while loop_result.action_taken == "retry_needed" and retries_made < 2:
    retries_made += 1
    retry_hint = loop_result.retry_context.to_hint()
    retry_result = await self._invoke_react(
        user_input, session_id, intent_ctx, trace_id, steps,
        suffix=f"\n[反思重试 #{retries_made}]\n{retry_hint}",
    )
    # 重新反思评估...
```

重试走同一套 `_invoke_react()`，只是注入反思反馈到 system prompt。

---

## ⑦ 记忆写入与回答边界控制

### 四层并行写入

```python
await self._memory.append_turn(session_id, "user", user_text)
await self._memory.append_turn(session_id, "assistant", answer)

# 内部并行旁路：
# L1 内存 dict（同步，始终写入）
# L3 PostgreSQL（主持久化）
# L2 Redis List（滑动窗口，TTL）
# L4 Milvus（语义向量，旁路写入）
# 各层 try/except 隔离，失败不阻塞其他层
```

### 回答边界控制

```python
answer = policy.with_boundary(content, user_text)

# 内部逻辑：
# if is_high_risk(query) → 返回 emergency_notice（"请立即拨打 120..."）
# elif disclaimer 不在 answer 中 → 追加 disclaimer（"以上内容仅供健康信息参考..."）
# else → 原样返回
```

---

## Plan-and-Execute 路径（对照）

当模式路由选择 `plan_execute` 时，流程变为：

```
PlannerAgent.plan() → 生成子任务列表
        ↓
逐个子任务调用 _invoke_react()（复用同一套 FC/正则循环）
        ↓
失败子任务 → Replanner 重新规划
        ↓
Synthesizer 汇总所有子任务结果
        ↓
反思闭环（同上）
```

Plan-and-Execute 和 ReAct 共享 `_invoke_react()`，区别只在**外层有没有 Planner 拆解任务**。

---

## 完整数据流

```
用户 query: "ARB药物副作用大吗，和ACEI相比哪个更适合我"
     │
① DomainPolicy.is_high_risk() → False（不含急诊关键词）
     │
② get_relevant(session_id, query, limit=6)
   → Redis → ["user: 高血压怎么治疗", "assistant: ...", ...]
   → Milvus → ["[历史记忆] 三个月前讨论过血压监测...", ...]
     │
③ IntentRecognizer.recognize(query, history)
   → LLM 分类 → IntentResult(intent="知识问答", confidence=0.88, slots={"compare_entities": ["ARB", "ACEI"]})
     │
④ ModeRouter.resolve(query, intent, history)
   → 规则匹配：知识问答 + 无时间敏感词 → ModeDecision(mode="react", source="rule")
     │
⑤ _invoke_react(mode="react")
   → FunctionCallAgent.run()
      Step 0: tool_call → rag_search("ARB ACEI 药物对比 副作用")
      Step 1: tool_call → web_search("ARB ACEI 最新临床对比")
      Step 2: 无 tool_calls → Final Answer
     │
⑥ ReflectionLoop.evaluate_and_improve()
   → quality_score=82 → action_taken="none" → 直接返回
     │
⑦ append_turn(user) + append_turn(assistant)
   → 四层并行写入
   policy.with_boundary(answer) → 追加免责声明
     │
返回最终答案 + trace_id + 6 个 steps + reflection_score=82
```
