# 通用 Agent/RAG 平台 — 心血管慢病默认 Demo

一个可插拔领域的 Agent/RAG 工程平台：用户上传任意领域的 PDF、TXT 或 Markdown 文档后，系统会完成解析、分块、混合检索和带引用回答。心血管慢病管理是默认 Demo，用来展示高风险领域的安全分流、回答边界和完整产品体验。

> 产品定位：**通用 Agent/RAG 底座 + 心血管默认领域插件**。通用底座不绑定医疗；医疗 Prompt、风险关键词和免责声明只在 cardiovascular 策略下启用。

---

## 核心特性

- **多范式 Agent**：`react` 与 `plan_execute`，按任务复杂度路由；Function Calling 可作为工具调用后端。
- **混合 RAG**：BM25（jieba）+ 向量检索（Milvus Lite）双路召回，RRF 融合，可选重排序和引用回答。
- **可插拔领域策略**：`general` 处理通用文档，`cardiovascular` 增加医疗安全提示；后续可扩展其他领域。
- **分层记忆**：Redis 短期状态 → PostgreSQL 持久化 → 内存降级；长期语义记忆使用向量存储。
- **工程健壮性**：LLM/Redis/Milvus 降级、熔断、意图识别双通道、Agent 步骤追踪和反思评分。

## 运行模式

### 最小本地模式

只需要 Python、一个 OpenAI-compatible API Key 和前端 Node 环境。PostgreSQL、Redis 或 Milvus Lite 不可用时，项目会按现有链路降级；Milvus Lite 默认使用本地 `milvus.db`，无法初始化时退化为内存向量库。

```bash
python -m venv .venv
.venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
copy .env.example .env
# 编辑 .env，至少填写 OPENAI_API_KEY
python -m uvicorn app.main:app --port 8001
```

### 完整 Docker Compose 模式

完整模式提供 PostgreSQL、Redis 和 Milvus standalone，适合演示持久化和独立向量服务；首次启动成本更高，不是本地快速体验的必选项。

```bash
docker compose up --build
```

PostgreSQL 负责业务数据和持久化记忆，Redis 负责短期会话与 TTL，VectorStore 负责文档向量检索；内存仅用于开发和故障降级。当前保留 Milvus Lite，未来可以在不改 RAG 业务层的前提下增加 pgvector 实现。

## 快速验证

```bash
curl http://localhost:8001/api/v1/health
curl -X POST http://localhost:8001/api/v1/chat/agent ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"帮我算 123*456\"}],\"agent_mode\":\"auto\",\"domain\":\"general\"}"
```

## 技术栈与文档

后端使用 Python 3.11+、FastAPI、Pydantic v2、SQLAlchemy async、LangGraph/LangChain 和 OpenAI-compatible API；检索使用 BM25、jieba、RRF、Milvus Lite；前端使用 React、TypeScript、Vite 和 Tailwind CSS。

- [项目全景文档](docs/File/项目全景文档.md)
- [项目工作流程举例](docs/File/项目工作流程举例.md)
- [评估报告-混合检索](docs/File/评估报告-混合检索.md)
- [面试问答](docs/File/面试问答20题.md)
- [秋招项目改进与架构决策](docs/File/秋招项目改进评估与架构决策.md)

## 测试

```bash
python -m pytest
ruff check app tests
```

测试数量和评测数字以当前环境实际运行结果为准；评测脚本、查询标注和历史实验报告位于 `scripts/` 与 `docs/File/`。

## 目录结构

```text
app/                 # FastAPI、Agent、RAG、领域策略、记忆与工具
frontend/            # React + TypeScript 前端
docs/File/           # 项目文档、评估与面试材料
scripts/             # RAG 评估脚本
tests/               # 离线单元测试
```

## License

MIT
