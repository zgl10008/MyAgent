// === 消息角色 ===
export type MessageRole = 'user' | 'assistant' | 'system';

// === 聊天消息 ===
export interface ChatMessage {
  role: MessageRole;
  content: string;
}

// === 聊天请求 ===
export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

// === 聊天响应（非流式） ===
export interface ChatResponse {
  id: string;
  model: string;
  content: string;
  trace_id: string | null;
  usage: Record<string, number> | null;
}

// === 流式消息块 ===
export interface StreamChunk {
  content?: string;
  trace_id?: string;
  done?: boolean;
  error?: string;
}

// === 文档信息 ===
export interface DocumentInfo {
  id: string;
  filename: string;
  mime_type: string | null;
  status: string;
  created_at: string | null;
}

// === 文档上传响应 ===
export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  message: string;
}

// === 健康检查响应 ===
export interface HealthResponse {
  status: string;
}

// === 就绪检查响应 ===
export interface ReadinessResponse {
  status: string;
  database: string;
  app_env: string;
}

// === 内部展示用消息 ===
export interface DisplayMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
}

// ==========================================================================
// Agent 编排扩展类型
// ==========================================================================

/** 意图识别结果 */
export interface IntentInfo {
  intent: string;
  confidence: number;
  sub_intent: string | null;
  rationale: string;
}

/** ReAct 步骤 */
export interface StepInfo {
  step: number;
  phase: string;
  thought: string | null;
  action: string | null;
  action_input: Record<string, unknown> | null;
  observation: string | null;
  final_answer: string | null;
  error: string | null;
}

/** Agent 对话请求 */
export interface AgentChatRequest {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_tokens?: number;
  conversation_id?: string;
  agent_mode?: 'auto' | 'react' | 'plan_execute';
  domain?: 'general' | 'cardiovascular';
}

/** Agent 对话响应 */
export interface AgentChatResponse {
  id: string;
  model: string;
  content: string;
  trace_id: string | null;
  usage: Record<string, number> | null;
  agent_mode: string;
  intent: IntentInfo | null;
  steps: StepInfo[];
  tool_calls: string[];
  reflection_score: number | null;
  degraded: boolean;
  /** Agent 通过 rag_search 检索到的文档上下文 */
  retrieved_contexts: RetrievalResult[];
  /** 反思扩展字段 */
  reflection_warnings: string[];
  hallucination_detected: boolean;
  completeness_notes: string;
  correction_applied: boolean;
}

/** Agent 运行状态（前端跟踪用） */
export type AgentRunStatus = 'idle' | 'thinking' | 'acting' | 'done' | 'error';

// ==========================================================================
// RAG 扩展类型
// ==========================================================================

/** 检索结果 */
export interface RetrievalResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
  source: string;
}

/** 引用标注 */
export interface Citation {
  index: number;
  result_id: string;
  snippet: string;
}

/** RAG 对话响应 */
export interface RAGChatResponse {
  answer: string;
  citations: Citation[];
  contexts: RetrievalResult[];
  model: string;
  total_chunks: number;
}

/** RAG 摄入响应 */
export interface RAGIngestResponse {
  chunk_count: number;
  message: string;
}

/** RAG 状态 */
export interface RAGStatusResponse {
  total_chunks: number;
  status: string;
}
