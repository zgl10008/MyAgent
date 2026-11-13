interface Props {
  usage: Record<string, number> | null;
  stepsCount: number;
  toolCalls: string[];
  reflectionScore: number | null;
  degraded: boolean;
  agentMode: string;
}

function totalTokens(usage: Record<string, number> | null): number | null {
  if (!usage) return null;
  if (typeof usage.total_tokens === 'number') return usage.total_tokens;
  const sum = Object.values(usage).reduce((a, b) => a + b, 0);
  return sum > 0 ? sum : null;
}

function scoreColor(score: number): string {
  if (score >= 80) return 'text-emerald-600';
  if (score >= 50) return 'text-amber-600';
  return 'text-red-600';
}

function Metric({ icon, label, value }: { icon: string; label: string; value: string | null }) {
  return (
    <div className="rounded-xl bg-slate-50 border border-slate-100 p-3">
      <p className="text-xs text-slate-400 mb-1">{icon} {label}</p>
      <p className="text-lg font-bold text-slate-800 leading-none">{value ?? '—'}</p>
    </div>
  );
}

export default function MetricsPanel({
  usage, stepsCount, toolCalls, reflectionScore, degraded, agentMode,
}: Props) {
  const tokens = totalTokens(usage);
  const modeLabel: Record<string, string> = {
    react: 'ReAct', plan_execute: 'Plan-Execute', auto: '自动', safety: '安全拦截', clarify: '澄清',
  };

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-emerald-600 uppercase tracking-wider">📊 执行指标</h3>
        <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
          degraded ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'
        }`}>
          {degraded ? '⚠️ 降级' : '● 正常'} · {modeLabel[agentMode] || agentMode}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Metric icon="🧠" label="推理 Token" value={tokens !== null ? tokens.toLocaleString() : null} />
        <Metric icon="🔢" label="执行步骤" value={stepsCount > 0 ? `${stepsCount} 步` : null} />
        <Metric icon="🔧" label="工具调用" value={toolCalls.length > 0 ? `${toolCalls.length} 次` : '0 次'} />
        <div className="rounded-xl bg-slate-50 border border-slate-100 p-3">
          <p className="text-xs text-slate-400 mb-1">✅ 质量评分</p>
          <p className={`text-lg font-bold leading-none ${reflectionScore !== null ? scoreColor(reflectionScore) : 'text-slate-400'}`}>
            {reflectionScore !== null ? `${reflectionScore}/100` : '—'}
          </p>
        </div>
      </div>
    </div>
  );
}
