import { useState, useRef } from 'react';

interface Props {
  onSend: (text: string) => void;
  disabled: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSend, disabled, placeholder }: Props) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed); setText('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleInput = () => {
    const el = textareaRef.current; if (!el) return;
    el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  };

  return (
    <div className="border-t border-emerald-200 bg-white px-4 py-3">
      <div className="flex items-end gap-2 max-w-4xl mx-auto">
        <textarea
          ref={textareaRef} value={text}
          onChange={(e) => setText(e.target.value)} onKeyDown={handleKeyDown} onInput={handleInput}
          placeholder={placeholder || '输入消息... (Enter 发送, Shift+Enter 换行)'}
          rows={1} disabled={disabled}
          className="flex-1 resize-none rounded-xl border border-slate-300 px-4 py-2.5 text-sm outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 disabled:bg-slate-50 disabled:cursor-not-allowed transition-colors"
        />
        <button
          onClick={handleSend} disabled={disabled || !text.trim()}
          className="px-5 py-2.5 bg-emerald-600 text-white rounded-xl text-sm font-medium hover:bg-emerald-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors shrink-0"
        >
          {disabled ? '分析中...' : '发送'}
        </button>
      </div>
    </div>
  );
}
