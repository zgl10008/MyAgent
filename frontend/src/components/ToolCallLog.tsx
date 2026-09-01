const medicalToolMap: Record<string, string> = {
  web_search: '🔍 查询医学资料',
  calculator: '🧮 计算健康指标',
  database_query: '🗄️ 查询健康记录',
};

interface Props {
  toolCalls: string[];
  reflectionScore: number | null;
  degraded: boolean;
  medical?: boolean;
  /** 反思扩展字段 */
  hallucinationDetected?: boolean;
  completenessNotes?: string;
  correctionApplied?: boolean;
  reflectionWarnings?: string[];
}

export default function ToolCallLog({
  toolCalls, reflectionScore, degraded, medical,
  hallucinationDetected, completenessNotes, correctionApplied, reflectionWarnings,
}: Props) {
  const hasContent =
    toolCalls.length > 0 ||
    reflectionScore !== null ||
    degraded ||
    hallucinationDetected ||
    correctionApplied;
  if (!hasContent) return null;
  const map = medical ? medicalToolMap : {};

  return (
    <div className="bg-white rounded-xl border border-emerald-200 p-4">
      <h3 className="text-xs font-semibold text-emerald-600 mb-2 uppercase tracking-wider">📋 执行摘要</h3>

      {toolCalls.length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-slate-400 mb-1">{medical ? '健康分析工具' : '调用的工具'}</p>
          <div className="flex flex-wrap gap-1.5">
            {toolCalls.map((tool) => (
              <span key={tool} className="inline-flex items-center gap-1 px-2 py-1 bg-emerald-50 text-emerald-700 rounded-md text-xs font-medium">
                {map[tool] || `🔧 ${tool}`}
              </span>
            ))}
          </div>
        </div>
      )}

      {reflectionScore !== null && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">{medical ? '安全评分' : '质量评分'}</span>
          <span className={`text-sm font-bold ${reflectionScore >= 80 ? 'text-emerald-600' : reflectionScore >= 50 ? 'text-amber-600' : 'text-red-600'}`}>
            {reflectionScore}/100
          </span>
        </div>
      )}

      {/* 反思警告 */}
      {reflectionWarnings && reflectionWarnings.length > 0 && (
        <div className="mt-2 space-y-1">
          {reflectionWarnings.map((w, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded">
              <span className="shrink-0">⚠️</span>
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* 疑似幻觉警告 */}
      {hallucinationDetected && (
        <div className="mt-2 flex items-start gap-1.5 text-xs text-red-700 bg-red-50 px-2 py-1 rounded">
          <span className="shrink-0">🚨</span>
          <span>检测到可能存在不准确信息，建议核实关键事实</span>
        </div>
      )}

      {/* 完整性备注 */}
      {completenessNotes && (
        <div className="mt-2 text-xs text-slate-500 bg-slate-50 px-2 py-1 rounded">
          {completenessNotes}
        </div>
      )}

      {/* 修正标记 */}
      {correctionApplied && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-blue-700 bg-blue-50 px-2 py-1 rounded">
          <span>🔧</span><span>回答已自动修正</span>
        </div>
      )}

      {degraded && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
          <span>⚠️</span><span>已降级运行</span>
        </div>
      )}
    </div>
  );
}
