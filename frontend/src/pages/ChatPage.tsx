import { useState, useCallback, useEffect } from 'react';
import type { AgentChatResponse, DisplayMessage, RAGChatResponse } from '../types';
import { agentChat } from '../api/chat';
import { getSession } from '../api/sessions';
import MessageList from '../components/MessageList';
import ChatInput from '../components/ChatInput';
import AgentSteps from '../components/AgentSteps';
import IntentBadge from '../components/IntentBadge';
import ToolCallLog from '../components/ToolCallLog';
import RAGContext from '../components/RAGContext';
import HealthPanel from '../components/HealthPanel';

let msgIdCounter = 0;
function nextId() { return `msg_${Date.now()}_${++msgIdCounter}`; }

type ChatMode = 'direct' | 'agent' | 'rag';

async function callRag(messages: { role: string; content: string }[]): Promise<RAGChatResponse> {
  const res = await fetch('/api/v1/chat/rag', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, mode: 'hybrid', top_k: 5 }),
  });
  if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail || `HTTP ${res.status}`); }
  return res.json();
}

async function ingestDocument(file: File): Promise<number> {
  const formData = new FormData(); formData.append('file', file);
  const res = await fetch('/api/v1/rag/ingest', { method: 'POST', body: formData });
  if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail || `HTTP ${res.status}`); }
  const data = await res.json(); return data.chunk_count;
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
  const [autoMode, setAutoMode] = useState(true);
  const [resolvedMode, setResolvedMode] = useState<ChatMode>('agent');
  const [agentResult, setAgentResult] = useState<AgentChatResponse | null>(null);
  const [agentSteps, setAgentSteps] = useState<AgentChatResponse['steps']>([]);
  const [agentLoading, setAgentLoading] = useState(false);
  const [ragResult, setRagResult] = useState<RAGChatResponse | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragChunks, setRagChunks] = useState(0);
  const [documents, setDocuments] = useState<{ filename: string; uploaded_at: number }[]>([]);

  const modeLabel: Record<ChatMode, { icon: string; label: string; desc: string }> = {
    direct: { icon: '💬', label: '直连', desc: '直接调用大模型' },
    agent:  { icon: '🤖', label: 'Agent', desc: '多步推理 · 工具调用 · 反思质检' },
    rag:    { icon: '📚', label: 'RAG', desc: '医学知识检索 · 指南引用' },
  };

  // 加载会话历史
  useEffect(() => {
    if (!sessionId) { setMessages([]); setAgentResult(null); setAgentSteps([]); setRagResult(null); return; }
    let cancelled = false;
    (async () => {
      try {
        const detail = await getSession(sessionId);
        if (cancelled) return;
        setMessages(detail.messages.map((m) => ({ id: nextId(), role: m.role as DisplayMessage['role'], content: m.content, timestamp: Date.now() })));
      } catch { /* */ }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  // 加载 RAG 状态和文档列表
  useEffect(() => {
    (async () => {
      try {
        const [statusRes, docsRes] = await Promise.all([
          fetch('/api/v1/rag/status'), fetch('/api/v1/rag/documents'),
        ]);
        setRagChunks((await statusRes.json()).total_chunks || 0);
        if (docsRes.ok) setDocuments(await docsRes.json());
      } catch { /* */ }
    })();
  }, []);

  // 自动选择模式（必须在 handleSend 之前定义）
  const resolveMode = useCallback((text: string, hasDocs: boolean): ChatMode => {
    if (hasDocs && /什么|如何|怎么|为什么|哪些|解释|含义|区别|建议|指导|用量|禁忌|副作用|说明书|指南/.test(text)) return 'rag';
    if (/今天|最近|最新|202\d|新闻|搜索|查一下|天气|计算|算/.test(text)) return 'agent';
    return 'agent';
  }, []);

  const handleSend = useCallback(async (text: string) => {
    const userMsg: DisplayMessage = { id: nextId(), role: 'user', content: text, timestamp: Date.now() };
    const assistantMsg: DisplayMessage = { id: nextId(), role: 'assistant', content: '', timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setLoading(true);
    const convId = sessionId || `session_${Date.now()}`;
    if (!sessionId) onSessionChange(convId);
    const actualMode = autoMode ? resolveMode(text, ragChunks > 0) : mode;
    setResolvedMode(actualMode);
    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    try {
      if (actualMode === 'agent') {
        setAgentLoading(true); setAgentSteps([]); setAgentResult(null);
        const resp = await agentChat({
          messages: [...history, { role: 'user' as const, content: text }],
          temperature: 0.7, agent_mode: 'react', conversation_id: convId,
        });
        setAgentResult(resp); setAgentSteps(resp.steps);
        setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: resp.content } : m));
      } else if (actualMode === 'rag') {
        setRagLoading(true); setRagResult(null);
        const resp = await callRag([...history, { role: 'user', content: text }]);
        setRagResult(resp);
        setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: resp.answer } : m));
      } else {
        const res = await fetch('/api/v1/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: [...history, { role: 'user', content: text }], temperature: 0.7 }),
        });
        const data = await res.json();
        setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: data.content } : m));
      }
      onSessionsChanged();
    } catch (err: any) {
      setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: `⚠️ 请求失败: ${err.message}` } : m));
    } finally {
      setLoading(false); setAgentLoading(false); setRagLoading(false);
    }
  }, [messages, mode, autoMode, ragChunks, sessionId, onSessionChange, onSessionsChanged, resolveMode]);

  const handleHealthUpload = useCallback(async (file: File) => {
    try {
      const count = await ingestDocument(file);
      setRagChunks((c) => c + count);
      setDocuments((prev) => [...prev, { filename: file.name, uploaded_at: Date.now() }]);
      setMessages((prev) => [...prev, { id: nextId(), role: 'system', content: `📄 已上传 "${file.name}"，知识库新增 ${count} 个文本块`, timestamp: Date.now() }]);
    } catch (err: any) { alert(`上传失败: ${err.message}`); }
  }, []);

  const intentLabelMap: Record<string, string> = {
    '知识问答': '📖 知识问答', '实时搜索': '🔍 实时查询', '数学计算': '🧮 数值计算',
    '文档管理': '📄 档案管理', '闲聊对话': '💬 日常交流',
  };

  return (
    <div className="flex h-full">
      <div className="w-60 border-r border-emerald-200 bg-white/70 overflow-y-auto p-3 shrink-0 space-y-3">
        <HealthPanel onUpload={handleHealthUpload} ragChunks={ragChunks} documents={documents} />
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="border-b border-emerald-200 px-6 py-3 bg-white flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-emerald-800">🩺 健康咨询</h2>
            <p className="text-xs text-slate-400 mt-0.5">描述您的身体感受或健康问题，Agent 将结合指南给出建议</p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
              resolvedMode === 'rag' ? 'bg-purple-100 text-purple-700' :
              resolvedMode === 'agent' ? 'bg-emerald-100 text-emerald-700' : 'bg-blue-100 text-blue-700'
            }`}>
              {autoMode ? '🤖 自动' : '🔧 手动'} · {modeLabel[resolvedMode].icon} {modeLabel[resolvedMode].label}
            </span>
            <button onClick={() => setAutoMode(!autoMode)}
              className={`text-xs px-2 py-1 rounded-md transition-colors ${autoMode ? 'text-slate-400 hover:text-slate-600' : 'text-emerald-600 font-medium'}`}>
              {autoMode ? '手动选择 ▸' : '恢复自动'}
            </button>
            {!autoMode && (
              <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5">
                {(['agent', 'rag', 'direct'] as const).map((m) => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${mode === m ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>
                    {modeLabel[m].icon} {modeLabel[m].label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <MessageList messages={messages} loading={loading} />
        <ChatInput onSend={handleSend} disabled={loading} placeholder="描述您的身体感受或健康问题..." />
      </div>

      <div className="w-80 border-l border-emerald-200 bg-white/70 overflow-y-auto p-4 shrink-0">
        {(resolvedMode || mode) === 'agent' && (
          <div className="space-y-4">
            {agentResult?.intent && <IntentBadge intent={agentResult.intent} labelMap={intentLabelMap} />}
            {agentResult && <ToolCallLog toolCalls={agentResult.tool_calls} reflectionScore={agentResult.reflection_score} degraded={agentResult.degraded} medical />}
            <AgentSteps steps={agentSteps} loading={agentLoading} medical />
            {!agentLoading && agentSteps.length === 0 && (
              <div className="text-center py-8"><p className="text-3xl mb-2">🫀</p><p className="text-sm text-slate-400">Agent 分析过程将在此展示</p></div>
            )}
          </div>
        )}
        {(resolvedMode || mode) === 'rag' && (
          <div className="space-y-4">
            <div className="bg-white rounded-xl border border-emerald-200 p-4">
              <h3 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">📚 医学知识库</h3>
              <div className="flex items-center gap-2"><span className="text-2xl">📚</span>
                <div><p className="text-sm font-medium text-slate-700">{ragChunks > 0 ? `${ragChunks} 个文本块` : '暂无内容'}</p>
                <p className="text-xs text-slate-400">{ragChunks > 0 ? '可进行指南知识问答' : '上传医学文档后开始'}</p></div>
              </div>
            </div>
            <RAGContext contexts={ragResult?.contexts || []} citations={ragResult?.citations || []} loading={ragLoading} />
          </div>
        )}
        {(resolvedMode || mode) === 'direct' && (
          <div className="text-center py-12"><p className="text-sm text-slate-400">直连模式无附加面板</p></div>
        )}
      </div>
    </div>
  );
}
