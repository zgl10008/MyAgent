import type { RetrievalResult, Citation } from '../types';

interface Props {
  contexts: RetrievalResult[];
  citations: Citation[];
  loading: boolean;
}

function SourceBadge({ source }: { source: string }) {
  const colors: Record<string, string> = {
    vector: 'bg-blue-100 text-blue-700',
    keyword: 'bg-green-100 text-green-700',
    hybrid: 'bg-purple-100 text-purple-700',
  };
  return (
    <span
      className={`text-xs px-1.5 py-0.5 rounded font-medium ${
        colors[source] || 'bg-slate-100 text-slate-600'
      }`}
    >
      {source}
    </span>
  );
}

export default function RAGContext({ contexts, citations, loading }: Props) {
  if (loading && contexts.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wider">
          检索上下文
        </h3>
        <div className="text-sm text-slate-400 animate-pulse">检索中...</div>
      </div>
    );
  }

  if (contexts.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <h3 className="text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wider">
        检索上下文
        <span className="ml-2 text-slate-400 font-normal normal-case">
          {contexts.length} 条
        </span>
      </h3>

      <div className="space-y-2">
        {contexts.map((ctx, idx) => {
          const citation = citations.find((c) => c.result_id === ctx.id);
          return (
            <div
              key={ctx.id || idx}
              className="border border-slate-100 rounded-lg p-2.5 hover:bg-slate-50 transition-colors"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <SourceBadge source={ctx.source} />
                <span className="text-xs text-slate-400">
                  分数: {ctx.score.toFixed(3)}
                </span>
                {citation && (
                  <span className="text-xs text-amber-600 font-medium">
                    引用 [{citation.index}]
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-600 leading-relaxed line-clamp-3">
                {ctx.content}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
