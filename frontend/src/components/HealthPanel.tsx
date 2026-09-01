import { useState, useEffect, useRef } from 'react';

interface Props {
  onUpload?: (file: File) => void;
  ragChunks: number;
  documents: { filename: string; uploaded_at: number }[];
}

export default function HealthPanel({ onUpload, ragChunks, documents }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 60000);
    return () => clearInterval(t);
  }, []);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && onUpload) onUpload(f);
    if (e.target) e.target.value = '';
  };

  return (
    <div className="space-y-4 text-sm">
      {/* 当前时间 */}
      <div className="bg-white rounded-xl border border-emerald-200 p-4">
        <p className="text-xs text-slate-400">{time.toLocaleDateString('zh-CN', { year:'numeric',month:'long',day:'numeric',weekday:'long' })}</p>
        <p className="text-2xl font-bold text-emerald-700 mt-1">
          {time.toLocaleTimeString('zh-CN', { hour:'2-digit', minute:'2-digit' })}
        </p>
      </div>

      {/* 上传医学文档 */}
      {onUpload && (
        <div className="bg-white rounded-xl border border-emerald-200 p-4">
          <h3 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
            📤 上传文档
          </h3>
          <input ref={fileRef} type="file" accept=".pdf,.txt,.md" className="hidden" onChange={handleFile} />
          <button
            onClick={() => fileRef.current?.click()}
            className="w-full py-2 text-xs rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 font-medium transition-colors"
          >
            + 上传文档
          </button>
          <p className="text-xs text-slate-400 mt-1">
            知识库: {ragChunks > 0 ? `${ragChunks} 个文本块` : '暂无内容'}
          </p>
        </div>
      )}

      {/* 已上传档案 */}
      {documents.length > 0 && (
        <div className="bg-white rounded-xl border border-emerald-200 p-4">
          <h3 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
            📑 健康档案
          </h3>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {documents.slice(0, 5).map((d, i) => (
              <p key={i} className="text-xs text-slate-600 truncate" title={d.filename}>
                📄 {d.filename}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* 安全提示 */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-3">
        <p className="text-xs text-amber-700 leading-relaxed">
          知识库支持任意领域文档；当前默认展示心血管安全策略。
        </p>
      </div>
    </div>
  );
}
