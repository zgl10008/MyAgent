import type { DisplayMessage } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Props {
  messages: DisplayMessage[];
  loading: boolean;
  onSuggest?: (text: string) => void;
}

const capabilities = [
  { icon: '🤖', title: '多范式 Agent', desc: 'ReAct / Plan-Execute 自动路由，含反思质检' },
  { icon: '📚', title: '混合检索 RAG', desc: 'BM25 + 向量 + 重排，回答带来源引用' },
  { icon: '🛡️', title: '医疗安全分流', desc: '风险分级与拒答门控，急诊症状即时拦截' },
  { icon: '🧠', title: '分层记忆', desc: '短期会话状态 + 长期语义记忆' },
];

const suggestions = [
  '高血压的诊断标准是什么？',
  '降压药有哪些常见副作用？',
  '低盐饮食具体应该怎么做？',
  '帮我计算 BMI：身高 175cm，体重 78kg',
];

export default function MessageList({ messages, loading, onSuggest }: Props) {
  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.length === 0 && !loading && (
        <div className="flex items-center justify-center h-full">
          <div className="max-w-2xl w-full text-center fade-in">
            <div className="text-5xl mb-4">🫀</div>
            <h2 className="text-xl font-bold text-slate-800">HeartGuard 心血管慢病管理 Agent</h2>
            <p className="text-sm text-slate-500 mt-1.5">
              上传心血管文档或直接提问，体验多范式 Agent 与混合检索 RAG 的完整链路
            </p>

            <div className="grid grid-cols-2 gap-3 mt-6 text-left">
              {capabilities.map((c) => (
                <div key={c.title} className="card p-3.5 hover:shadow-md transition-shadow">
                  <p className="text-lg mb-1">{c.icon}</p>
                  <p className="text-sm font-semibold text-slate-700">{c.title}</p>
                  <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{c.desc}</p>
                </div>
              ))}
            </div>

            <div className="mt-6">
              <p className="text-xs text-slate-400 mb-2">试试这样问</p>
              <div className="flex flex-wrap justify-center gap-2">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => onSuggest?.(s)}
                    className="text-xs px-3 py-1.5 rounded-full border border-emerald-200 text-emerald-700 bg-white hover:bg-emerald-50 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} fade-in`}
        >
          {msg.role === 'assistant' && (
            <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center text-sm mr-2 shrink-0 self-end mb-1">
              🫀
            </div>
          )}
          <div
            className={`max-w-[75%] rounded-2xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-emerald-600 text-white'
                : 'bg-white text-slate-800 border border-slate-200 shadow-sm'
            } ${msg.role === 'system' ? 'bg-amber-50 text-amber-700 border border-amber-200 max-w-full' : ''}`}
          >
            {msg.role === 'assistant' ? (
              <div className="markdown-body text-sm leading-relaxed">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
            )}
          </div>
        </div>
      ))}
      {loading && (
        <div className="flex justify-start fade-in">
          <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center text-sm mr-2 shrink-0 self-end mb-1">
            🫀
          </div>
          <div className="bg-white border border-slate-200 shadow-sm rounded-2xl px-4 py-3">
            <div className="flex gap-1.5">
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" />
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.1s]" />
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.2s]" />
            </div>
          </div>
        </div>
      )}
      <div ref={(el) => el?.scrollIntoView({ behavior: 'smooth' })} />
    </div>
  );
}
