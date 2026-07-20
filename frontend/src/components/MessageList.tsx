import type { DisplayMessage } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Props {
  messages: DisplayMessage[];
  loading: boolean;
}

export default function MessageList({ messages, loading }: Props) {
  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.length === 0 && !loading && (
        <div className="flex items-center justify-center h-full text-slate-400">
          <div className="text-center">
            <div className="text-4xl mb-3">🤖</div>
            <p className="text-lg font-medium text-slate-500">开始对话</p>
            <p className="text-sm mt-1">发送一条消息，体验 AI Agent 的能力</p>
          </div>
        </div>
      )}
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} fade-in`}
        >
          <div
            className={`max-w-[75%] rounded-2xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-100 text-slate-800'
            }`}
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
          <div className="bg-slate-100 rounded-2xl px-4 py-3">
            <div className="flex gap-1.5">
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.1s]" />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]" />
            </div>
          </div>
        </div>
      )}
      <div ref={(el) => el?.scrollIntoView({ behavior: 'smooth' })} />
    </div>
  );
}
