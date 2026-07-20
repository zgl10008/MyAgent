interface Props {
  toolCalls: string[];
  reflectionScore: number | null;
  degraded: boolean;
}

export default function ToolCallLog({ toolCalls, reflectionScore, degraded }: Props) {
  if (toolCalls.length === 0 && reflectionScore === null && !degraded) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <h3 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
        执行摘要
      </h3>

      {toolCalls.length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-slate-400 mb-1">调用的工具</p>
          <div className="flex flex-wrap gap-1.5">
            {toolCalls.map((tool) => (
              <span
                key={tool}
                className="inline-flex items-center gap-1 px-2 py-1 bg-slate-100 text-slate-700 rounded-md text-xs font-medium"
              >
                🔧 {tool}
              </span>
            ))}
          </div>
        </div>
      )}

      {reflectionScore !== null && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">质量评分</span>
          <span
            className={`text-sm font-bold ${
              reflectionScore >= 80
                ? 'text-green-600'
                : reflectionScore >= 50
                ? 'text-yellow-600'
                : 'text-red-600'
            }`}
          >
            {reflectionScore}/100
          </span>
        </div>
      )}

      {degraded && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
          <span>⚠️</span>
          <span>已降级运行（Plan 失败回退 ReAct）</span>
        </div>
      )}
    </div>
  );
}
