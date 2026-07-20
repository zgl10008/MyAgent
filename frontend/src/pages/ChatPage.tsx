import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  AgentChatResponse,
  ChatMessage,
  DisplayMessage,
  RAGChatResponse,
} from '../types';
import { agentChat } from '../api/chat';
import { getSession } from '../api/sessions';
import MessageList from '../components/MessageList';
import ChatInput from '../components/ChatInput';
import AgentSteps from '../components/AgentSteps';
import IntentBadge from '../components/IntentBadge';
import ToolCallLog from '../components/ToolCallLog';
import RAGContext from '../components/RAGContext';

let msgIdCounter = 0;
function nextId() {
  return `msg_${Date.now()}_${++msgIdCounter}`;
}

type ChatMode = 'direct' | 'agent' | 'rag';

/** 调用 RAG API */
async function callRag(
  messages: { role: string; content: string }[]
): Promise<RAGChatResponse> {
  const res = await fetch('/api/v1/chat/rag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, mode: 'hybrid', top_k: 5 }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/** 上传文档到 RAG 知识库 */
async function ingestDocument(file: File): Promise<number> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch('/api/v1/rag/ingest', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return data.chunk_count;
}

interface Props {
  sessionId: string | null;
  onSessionChange: (id: string | null) => void;
  onSessionsChanged: () => void;
}

export default function ChatPage({ sessionId, onSessionChange, onSessionsChanged }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<ChatMode>('agent');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const initRef = useRef(false);

  // Agent 状态
  const [agentResult, setAgentResult] = useState<AgentChatResponse | null>(null);
  const [agentSteps, setAgentSteps] = useState<AgentChatResponse['steps']>([]);
  const [agentLoading, setAgentLoading] = useState(false);

  // RAG 状态
  const [ragResult, setRagResult] = useState<RAGChatResponse | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragChunks, setRagChunks] = useState(0);

  const modeLabel: Record<ChatMode, { icon: string; label: string; desc: string }> = {
    direct: { icon: '💬', label: '直连', desc: '直接调用大模型' },
    agent: { icon: '🤖', label: 'Agent', desc: 'ReAct 编排 · 工具调用 · 反思' },
    rag: { icon: '📚', label: 'RAG', desc: '知识库检索 · 引用增强' },
  };

  // 加载会话历史
  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      setAgentResult(null);
      setAgentSteps([]);
      setRagResult(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const detail = await getSession(sessionId);
        if (cancelled) return;
        setMessages(
          detail.messages.map((m) => ({
            id: nextId(),
            role: m.role as DisplayMessage['role'],
            content: m.content,
            timestamp: Date.now(),
          }))
        );
      } catch {
        // session might not exist anymore
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  // 加载 RAG 知识库状态
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api/v1/rag/status');
        const data = await res.json();
        setRagChunks(data.total_chunks || 0);
      } catch {
        // ignore
      }
    })();
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg: DisplayMessage = {
        id: nextId(),
        role: 'user',
        content: text,
        timestamp: Date.now(),
      };
      const assistantMsg: DisplayMessage = {
        id: nextId(),
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setLoading(true);

      // 确保有会话 ID
      const convId = sessionId || `session_${Date.now()}`;
      if (!sessionId) {
        onSessionChange(convId);
      }

      const history = messages.map((m) => ({ role: m.role, content: m.content }));

      try {
        if (mode === 'agent') {
          setAgentLoading(true);
          setAgentSteps([]);
          setAgentResult(null);

          const resp = await agentChat({
            messages: [...history, { role: 'user' as const, content: text }],
            temperature: 0.7,
            agent_mode: 'react',
            conversation_id: convId,
          });

          setAgentResult(resp);
          setAgentSteps(resp.steps);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id ? { ...m, content: resp.content } : m
            )
          );
        } else if (mode === 'rag') {
          setRagLoading(true);
          setRagResult(null);

          const resp = await callRag([
            ...history,
            { role: 'user', content: text },
          ]);

          setRagResult(resp);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id ? { ...m, content: resp.answer } : m
            )
          );
        } else {
          // direct
          const res = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [...history, { role: 'user', content: text }],
              temperature: 0.7,
            }),
          });
          const data = await res.json();
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id ? { ...m, content: data.content } : m
            )
          );
        }

        // 通知侧边栏刷新会话列表
        onSessionsChanged();
      } catch (err: any) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: `⚠️ 请求失败: ${err.message}` }
              : m
          )
        );
      } finally {
        setLoading(false);
        setAgentLoading(false);
        setRagLoading(false);
      }
    },
    [messages, mode, sessionId, onSessionChange, onSessionsChanged]
  );

  // 上传文档到 RAG
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const count = await ingestDocument(file);
      setRagChunks((c) => c + count);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'system',
          content: `📄 已上传 "${file.name}"，知识库新增 ${count} 个文本块`,
          timestamp: Date.now(),
        },
      ]);
    } catch (err: any) {
      alert(`上传失败: ${err.message}`);
    }
    if (e.target) e.target.value = '';
  };

  return (
    <div className="flex h-full">
      {/* 左侧：聊天区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶栏 */}
        <div className="border-b border-slate-200 px-6 py-3 bg-white flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-800">
              {modeLabel[mode].icon} {modeLabel[mode].label}
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">{modeLabel[mode].desc}</p>
          </div>
          <div className="flex items-center gap-2">
            {/* 模式切换 */}
            <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5">
              {(['direct', 'agent', 'rag'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors ${
                    mode === m
                      ? 'bg-white text-slate-800 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  {modeLabel[m].icon} {modeLabel[m].label}
                </button>
              ))}
            </div>
            {/* RAG 上传按钮 */}
            {mode === 'rag' && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.pdf,.md"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3 py-1.5 text-xs rounded-lg bg-emerald-50 text-emerald-600 border border-emerald-200 hover:bg-emerald-100 font-medium transition-colors"
                >
                  + 上传文档
                </button>
              </>
            )}
          </div>
        </div>

        <MessageList messages={messages} loading={loading} />
        <ChatInput onSend={handleSend} disabled={loading} />
      </div>

      {/* 右侧面板 */}
      <div className="w-80 border-l border-slate-200 bg-slate-50/50 overflow-y-auto p-4 shrink-0">
        {mode === 'agent' && (
          <div className="space-y-4">
            {agentResult?.intent && <IntentBadge intent={agentResult.intent} />}
            {agentResult && (
              <ToolCallLog
                toolCalls={agentResult.tool_calls}
                reflectionScore={agentResult.reflection_score}
                degraded={agentResult.degraded}
              />
            )}
            <AgentSteps steps={agentSteps} loading={agentLoading} />
            {!agentLoading && agentSteps.length === 0 && (
              <div className="text-center py-8">
                <p className="text-sm text-slate-400">Agent 思考过程将在此展示</p>
              </div>
            )}
          </div>
        )}

        {mode === 'rag' && (
          <div className="space-y-4">
            {/* 知识库状态 */}
            <div className="bg-white rounded-xl border border-slate-200 p-4">
              <h3 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
                知识库
              </h3>
              <div className="flex items-center gap-2">
                <span className="text-2xl">📚</span>
                <div>
                  <p className="text-sm font-medium text-slate-700">
                    {ragChunks > 0
                      ? `${ragChunks} 个文本块`
                      : '暂无内容'}
                  </p>
                  <p className="text-xs text-slate-400">
                    {ragChunks > 0 ? '可进行知识问答' : '上传文档后开始 RAG'}
                  </p>
                </div>
              </div>
            </div>
            <RAGContext
              contexts={ragResult?.contexts || []}
              citations={ragResult?.citations || []}
              loading={ragLoading}
            />
          </div>
        )}

        {mode === 'direct' && (
          <div className="text-center py-12">
            <p className="text-sm text-slate-400">直连模式无附加面板</p>
          </div>
        )}
      </div>
    </div>
  );
}
