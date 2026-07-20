# -*- coding: utf-8 -*-
"""API 请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import MessageRole


class ChatMessage(BaseModel):
    """单条对话消息。"""

    role: str = Field(description="角色：system | user | assistant")
    content: str = Field(description="文本内容")


class ChatRequest(BaseModel):
    """对话请求。"""

    messages: list[ChatMessage] = Field(min_length=1, description="OpenAI 风格消息列表")
    model: str | None = Field(default=None, description="优先使用的模型 ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    conversation_id: str | None = Field(default=None, description="可选会话 ID")


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    id: str = Field(description="响应 ID")
    model: str = Field(description="实际使用的模型")
    content: str = Field(description="助手回复正文")
    trace_id: str | None = Field(default=None, description="链路追踪 ID")
    usage: dict[str, Any] | None = Field(default=None, description="Token 用量")


class DocumentUploadResponse(BaseModel):
    """文档上传响应。"""

    document_id: str
    filename: str
    status: str
    chunk_count: int = 0
    message: str = "ok"


class DocumentInfo(BaseModel):
    """文档列表项。"""

    id: str
    filename: str
    mime_type: str | None = None
    status: str
    created_at: str | None = None


class DocumentUploadRequest(BaseModel):
    """文档上传附加元数据（可选）。"""

    tags: list[str] = Field(default_factory=list)


class Message(BaseModel):
    """对话消息（记忆/RAG 内部使用，含角色枚举）。"""

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """长期记忆召回条目。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """短期 + 长期记忆合并上下文。"""

    session_id: str
    short_term_messages: list[Message]
    long_term_items: list[MemoryItem]


class RetrievalResult(BaseModel):
    """检索单条结果。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "vector"


class Citation(BaseModel):
    """答案中的引用标注。"""

    index: int
    result_id: str
    snippet: str


class RAGResponse(BaseModel):
    """RAG 生成结果。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    raw_contexts: list[RetrievalResult] = Field(default_factory=list)
    model: str | None = None


# ==============================================================================
# Agent 编排扩展模型
# ==============================================================================


class IntentInfo(BaseModel):
    """意图识别结果（API 对外版本）。"""

    intent: str = Field(description="主意图标签")
    confidence: float = Field(description="置信度 0~1")
    sub_intent: str | None = Field(default=None, description="子意图")
    rationale: str = Field(default="", description="识别理由")


class StepInfo(BaseModel):
    """ReAct 步骤信息。"""

    step: int = Field(description="步骤序号")
    phase: str = Field(default="react", description="阶段标识")
    thought: str | None = Field(default=None, description="模型推理内容")
    action: str | None = Field(default=None, description="调用的工具名")
    action_input: dict[str, Any] | None = Field(default=None, description="工具参数")
    observation: str | None = Field(default=None, description="工具返回结果")
    final_answer: str | None = Field(default=None, description="最终答案")
    error: str | None = Field(default=None, description="步骤错误信息")


class AgentChatRequest(BaseModel):
    """Agent 对话请求（扩展版）。"""

    messages: list[ChatMessage] = Field(min_length=1, description="OpenAI 风格消息列表")
    model: str | None = Field(default=None, description="优先使用的模型 ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    conversation_id: str | None = Field(default=None, description="会话 ID")
    agent_mode: str = Field(default="react", description="Agent 模式: react / plan_execute")


class AgentChatResponse(BaseModel):
    """Agent 对话响应（含步骤追踪）。"""

    id: str = Field(description="响应 ID")
    model: str = Field(description="实际使用的模型")
    content: str = Field(description="助手回复正文")
    trace_id: str | None = Field(default=None, description="链路追踪 ID")
    usage: dict[str, Any] | None = Field(default=None, description="Token 用量")

    # Agent 扩展字段
    agent_mode: str = Field(default="react", description="使用的 Agent 模式")
    intent: IntentInfo | None = Field(default=None, description="意图识别结果")
    steps: list[StepInfo] = Field(default_factory=list, description="Agent 执行步骤")
    tool_calls: list[str] = Field(default_factory=list, description="本次调用的工具列表")
    reflection_score: int | None = Field(default=None, description="反思质量评分 0-100")
    degraded: bool = Field(default=False, description="是否降级")


# ==============================================================================
# RAG 扩展模型
# ==============================================================================


class RAGChatRequest(BaseModel):
    """RAG 对话请求。"""

    messages: list[ChatMessage] = Field(min_length=1, description="对话历史+当前问题")
    query: str | None = Field(default=None, description="显式检索查询（为空则取最后一条消息）")
    mode: str = Field(default="hybrid", description="检索模式: vector / keyword / hybrid")
    top_k: int = Field(default=5, ge=1, le=20, description="检索数量")


class RAGChatResponse(BaseModel):
    """RAG 对话响应。"""

    answer: str = Field(description="生成的回答")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    contexts: list[RetrievalResult] = Field(default_factory=list, description="检索上下文")
    model: str = Field(default="", description="使用的模型")
    total_chunks: int = Field(default=0, description="知识库总块数")


class RAGIngestResponse(BaseModel):
    """RAG 文档摄入响应。"""

    chunk_count: int = Field(description="分块数量")
    message: str = Field(default="ok", description="处理结果")
