# 短期记忆 Redis 实现详解

> 配套 [面试版架构图.md](面试版架构图.md) 的「三层记忆架构」使用。
> 代码位置：[app/core/memory/short_term.py](../../app/core/memory/short_term.py)

短期记忆用 **Redis List** 存消息序列，超量时「旧的压成摘要、近的保留原文」，把单会话上下文始终压在 4000 token / 20 条以内。本文只讲 Redis 层面的具体数据结构。

---

## 1. 类型与 Key 命名

```
类型   : Redis List
Key    : stm:session:{session_id}
元素   : 一条消息的 JSON 字符串
顺序   : 时间顺序（RPUSH 追加，索引 0 = 最早）
```

Key 前缀在 `short_term.py:44` 写死为 `stm:session:`，加 session_id 拼成完整 key。

## 2. 元素格式（每个元素是一条消息）

```json
{"role": "user", "content": "我最近血压有点高怎么办", "metadata": {}}
```

由 `_serialize`（`short_term.py:59-65`）序列化：
- `role`：枚举值（user / assistant / system）
- `content`：消息正文
- `metadata`：透传调用方附加信息

## 3. 一个会话在 Redis 里的实际样子

session_id 为 `abc123`，聊了 3 轮后，`stm:session:abc123` 这个 List 有 6 个元素：

```
索引    元素内容
[0]  {"role":"user","content":"我血压有点高","metadata":{}}
[1]  {"role":"assistant","content":"您最近有在服药吗？","metadata":{}}
[2]  {"role":"user","content":"在吃氨氯地平","metadata":{}}
[3]  {"role":"assistant","content":"氨氯地平是常用降压药…","metadata":{}}
[4]  {"role":"user","content":"饮食上要注意什么","metadata":{}}
[5]  {"role":"assistant","content":"建议低盐饮食，每日<5g…","metadata":{}}
```

## 4. 命令 ↔ 代码 对照

| Redis 命令 | 代码位置 | 作用 |
|---|---|---|
| `RPUSH key element` | `short_term.py:113` | 追加一条消息 |
| `LRANGE key 0 -1` | `short_term.py:83` | 读全部（按时间序） |
| `LLEN key` | `short_term.py:130` | 查条数（判断是否超窗） |
| `DEL key` | `short_term.py:159` | 压缩重写前先清空 |
| `DEL key` | `short_term.py:104` | 删除会话时清理 |

## 5. 压缩后的布局变化

假设聊了 30 条，触发压缩（超 20 条）。压缩后 List 变成 **11 个元素**：

```
索引    内容
[0]  {"role":"system","content":"[历史摘要]\n用户咨询高血压…","metadata":{"compressed_from":20}}
[1]  {"role":"assistant","content":"…"}   ← 尾部保留的原文
[2]  {"role":"user","content":"…"}
 …
[10] {"role":"assistant","content":"…"}
```

关键：**索引 0 永远是摘要（SYSTEM 角色，`[历史摘要]` 前缀），后面跟着保留的尾部 10 条原文**。`metadata.compressed_from` 记录这一条摘要「吃掉」了原来多少条消息，可追溯。

## 6. 短期记忆没有设 TTL

`ShortTermMemory` 里**没有 `EXPIRE`**——数据会一直留在 Redis，直到被压缩或显式删除。

所以这里的「短期」指的不是「存活时间短」（TTL），而是「**窗口短**」——靠滑窗 + 压缩把单会话上下文始终压在 4000 token / 20 条以内。

**追问标准答法**：

> 不会无限增长——单会话的 List 靠压缩控制在 20 条/4000 token 内；但确实不设 TTL，删除靠会话删除时显式 `DEL`。如果会话量大、担心内存，可以加 TTL 兜底，或压缩到一定次数后把摘要落 PostgreSQL。

## 7. 三个面试追问点

**① 为什么摘要要前置成 SYSTEM 消息？**
摘要代表「更早的历史背景」，放在列表最前面，LLM 看到的是「背景摘要 → 最近原文」的自然顺序；用 SYSTEM 角色区分「压缩过的历史」和「原始对话」。

**② 压缩重写是 `delete` 再 `rpush`，中间崩溃会丢数据吗？**
会，先删后写不是原子的。单进程场景风险低；要加固可用 Lua 脚本把「删+写」原子化，或先写新 key 再 rename 覆盖。主动提这个坑比藏着更显水平。

**③ 4000 token 和 20 条哪个先触发？**
看内容——短消息堆到 20 条先触发（条数）；长消息（用户贴大段文本）可能 5、6 条就撞 4000 token。所以需要**双阈值**，只靠条数会漏掉「单条超长」的场景。
