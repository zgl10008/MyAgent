# 问题记录：SerpAPI 搜索超时（国内网络不可达 + 代理配置未生效）

> 记录日期：2026-08-10
> 涉及模块：`app/core/tools/builtin/search.py`、`app/core/agent_service.py`、`app/config.py`、`.env`
> 状态：已修复并补充回归测试

---

## 一、现象

Agent 调用 `web_search` 工具时出现：

```
2026-08-10 17:48:55.202 | WARNING | app.core.tools.builtin.search:_search_serpapi:112 - SerpAPI 连接超时 query=2026年世界杯 赛程 举办地 夺冠热门
```

搜索功能返回"搜索超时，无法连接 SerpAPI"。

---

## 二、定位过程

1. `WebSearchTool` 默认超时 8 秒，由 `agent_service.py` 构造时固定传入。
2. 本机 `SEARCH_PROXY`、`HTTP_PROXY`、`HTTPS_PROXY` 环境变量全部为空，`.env` 中也未配置代理。
3. `serpapi.com` 在国内直连通常不可达，8 秒内无法建立连接，命中 `httpx.ConnectTimeout` 分支。
4. **附带发现的配置 bug**：工具代码里提示"在 `.env` 中设置 `SEARCH_PROXY`"，但 `WebSearchTool` 读取的是 `os.environ`；而 pydantic-settings 只会把 `.env` 的值读进 `Settings` 字段，**不会写入 `os.environ`**。因此即使按提示在 `.env` 里加了 `SEARCH_PROXY`，工具也看不到，配置了也不生效。

---

## 三、根因

- 直接原因：网络层不可达（国内访问 SerpAPI 需代理），且没有配置任何代理。
- 配置断点：`.env` 与工具读取的 `os.environ` 不通，代理配置路径是断的。
- 次要因素：超时时间写死在调用方，无法按网络情况调整；读取超时（非连接超时）没有专门兜底提示。

---

## 四、修复内容

### 1. `app/config.py`

- 新增 `search_proxy`（代理地址）与 `search_timeout`（超时秒数，默认 8.0）配置项。

### 2. `app/core/tools/builtin/search.py`

- `WebSearchTool` 新增 `proxy` 构造参数。
- 新增 `_resolve_proxy()`，每次调用时解析代理，优先级：**显式传入 > SEARCH_PROXY > HTTP_PROXY > HTTPS_PROXY**（此前只在 `__init__` 读一次环境变量）。
- 新增 `httpx.TimeoutException` 兜底分支，区分"连接超时"与"读取超时"，提示对应解法。
- 错误提示文案明确"改完 `.env` 需要重启服务"。

### 3. `app/core/agent_service.py`

- 构造 `WebSearchTool` 时注入 `settings.search_proxy` 与 `settings.search_timeout`。

### 4. `.env` / `.env.example`

- 在 SerpAPI 配置段补充 `SEARCH_PROXY`、`SEARCH_TIMEOUT` 注释模板。

### 5. 测试

- 新增 `tests/test_web_search_tool.py`（6 个用例）：显式代理优先于环境变量、环境变量兜底、无代理返回 None、空 query 报错、无 Key 返回提示、有 Key 时正确分发到 SerpAPI。

---

## 五、验证

- 完整测试套件：**76 passed**（原 70 + 新增 6）。

---

## 六、仍需用户操作

代码修复只解决了"配置了能生效"，无法替你解决网络可达性。要让搜索真正可用，二选一：

1. 在 `.env` 配置可用代理并重启服务：
   ```env
   SEARCH_PROXY=http://127.0.0.1:7890
   ```
2. 若走系统代理（如 Clash 已开 TUN/系统代理），直接把 `SEARCH_PROXY` 指向本地代理端口即可。

---

## 七、遗留与后续建议

1. 若始终无法使用代理，可增加国内可达的搜索后端兜底（如必应搜索 HTML 接口），SerpAPI 失败时自动切换。
2. 可考虑把代理读取改为每次请求实时解析（当前已支持环境变量变化，但 `.env` 改动仍需重启）。

---

## 八、最终方案（2026-08-10）：弃用 SerpAPI，改用百度搜索

实测代理节点对 SerpAPI 的 TLS 握手持续被风控（换节点后连 google 都不稳定），
且 SerpAPI 需要付费 + 代理，与"国内直连、免费、免 Key"的目标不符，决定**彻底弃用 SerpAPI**。

### 改动内容

- `app/core/tools/builtin/search.py`：重写为**百度网页搜索解析**（`https://www.baidu.com/s`），
  国内直连、无需 API Key、无需代理；解析标题/摘要/链接，检测"百度安全验证"并给出友好提示。
- `app/config.py`：移除 `serpapi_api_key`、`search_proxy` 配置项，保留 `search_timeout`。
- `app/core/agent_service.py`：改为 `WebSearchTool(timeout=settings.search_timeout)`。
- `.env` / `.env.example`：移除 `SERPAPI_API_KEY`、`SEARCH_PROXY`，保留 `SEARCH_TIMEOUT`。
- `tests/test_web_search_tool.py`：重写为百度用例（解析、验证码拦截、超时降级、空查询校验）。

### 已知取舍

- 百度没有面向普通开发者的免费搜索 API，采用 HTML 解析：页面结构一旦改版需要同步维护。
- 请求过于频繁会触发"百度安全验证"，Agent 低频调用场景影响不大。
- 中文结果质量好；英文、实时性较强的信息覆盖一般，必要时可后续再加国内可达的兜底后端。

### 遗留说明

`docs/File/` 等历史学习文档仍描述 SerpAPI 架构，属于源项目记录，本次未同步改写；
如需保持文档与代码一致，后续可统一更新。
