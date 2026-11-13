"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。"""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 限制底层 BLAS / 数学库线程数，避免 sentence-transformers + torch + OpenBLAS
# 线程过载（每线程各自申请工作内存）导致的 "OpenBLAS memory allocation failed"。
# 必须在 numpy/torch 首次初始化前设置（config 是大多数入口最先导入的模块）。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-ai-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API Base",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="默认对话模型")

    database_url: str = Field(
        default="",
        description="SQLAlchemy 异步数据库 URL（推荐 postgresql+asyncpg）；留空则跳过数据库初始化，走内存降级",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    milvus_db_path: str = Field(
        default="milvus.db",
        description="Milvus Lite 本地数据库文件路径",
    )
    milvus_collection_name: str = Field(
        default="rag_chunks",
        description="RAG 向量集合名",
    )

    search_timeout: float = Field(
        default=8.0,
        description="网络搜索（百度）请求超时（秒）",
    )

    mock_web_search: bool = Field(
        default=False,
        description="评测时 mock 百度搜索：秒回占位结果、不真打百度，避免搜索超时拖慢评测",
    )

    hf_endpoint: str = Field(
        default="https://hf-mirror.com",
        description="HuggingFace 镜像（国内加速）",
    )

    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="语义向量模型（中文/多语言检索推荐 bge-m3，1024 维；轻量可换 all-MiniLM-L6-v2，384 维）",
    )
    embedding_device: str = Field(
        default="",
        description="向量模型设备（cuda / cpu，留空自动检测，有 GPU 时优先 cuda）",
    )

    # Cross-Encoder 精排（默认开启）：RRF 召回后对候选集重排序，提升精度。
    # 首次启用会加载 cross-encoder 模型，失败自动降级为原始召回顺序。
    # 想关掉可设 RERANK_ENABLED=false（生产按需开关，精排会增加单次检索延迟）。
    rerank_enabled: bool = Field(
        default=True,
        description="是否启用 cross-encoder 精排（两阶段检索第二阶段）",
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-Encoder 重排序模型名或本地路径（中文检索需用多语言模型，如 bge-reranker-v2-m3；英文可换 ms-marco-MiniLM）",
    )
    rerank_candidate_k: int = Field(
        default=50,
        description="精排前召回的候选数量（cross-encoder 只对候选集打分）。"
        "默认 50 用于保留召回覆盖率；延迟敏感或显存有限时可降低到 20。",
    )
    rerank_batch_size: int = Field(
        default=4,
        ge=1,
        description="Cross-Encoder 单次 GPU 推理的候选对数量；显存有限时降低此值以避免长文档精排卡顿",
    )
    rerank_max_length: int = Field(
        default=384,
        ge=64,
        description="Cross-Encoder 输入的最大 token 长度，用于限制长候选的显存与推理开销",
    )
    rerank_device: str = Field(
        default="",
        description="Cross-Encoder 设备（cuda / cpu / mps，留空自动选择）",
    )

    # 库外拒答门控：top-1 精排相关度低于该阈值时，判定知识库无相关内容，返回拒答
    # 而非硬答（避免库外问题幻觉）。None=关闭（默认）。阈值只对 cross-encoder 精排分
    # （[0,1] 越大越相关）有意义，故仅 rerank 开启时生效。推荐值由
    # scripts/calibrate_refuse_threshold.py 在评测集上校准得到。
    rag_refuse_threshold: float | None = Field(
        default=None,
        description="库外拒答门控阈值（None=关闭；精排 top-1 分低于此值即拒答）",
    )

    safety_risk_classifier_enabled: bool = Field(
        default=True,
        description="医疗安全门是否启用 LLM 二次确认（区分正在发作/询问知识/既往史，降低关键词误拦）",
    )

    # 编排器开关：默认全开（生产行为），评测 A/B 消融时通过 env 关闭。
    # A/B 消融只变「重规划」与「反思」两项，其余（FC、失败降级）两组保持一致。
    agent_use_function_calling: bool = Field(
        default=True,
        description="ReAct/Plan 子任务是否走 Function Calling 结构化后端（False=纯文本解析）",
    )
    agent_enable_reflection: bool = Field(
        default=True,
        description="是否启用反思重试（质量低于阈值时触发自我反思再回答）",
    )
    agent_fallback_react_on_plan_failure: bool = Field(
        default=True,
        description="Plan-and-Execute 失败时是否降级回 ReAct",
    )
    agent_max_replan_attempts: int = Field(
        default=2,
        ge=0,
        description="Plan-and-Execute 子任务失败时的最大重规划次数（0=不重规划）",
    )

    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()
