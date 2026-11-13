import type { StepInfo } from '../types';

// 医疗场景步骤中文映射
const medicalPhaseMap: Record<string, string> = {
  react: '🔄 多步分析',
  execute: '⚙️ 执行中',
  replan: '🔁 重新规划',
  llm: '🧠 模型调用',
};
const medicalActionMap: Record<string, string> = {
  web_search: '🔍 查询医学资料',
  calculator: '🧮 计算健康指标',
  database_query: '🗄️ 查询健康记录',
};

function phaseLabel(phase: string, medical?: boolean): string {
  return medical ? (medicalPhaseMap[phase] || '🔄 多步分析') : (phase || '思考');
}

function ThoughtCard({ thought, medical }: { thought: string; medical?: boolean }) {
  if (!thought) return null;
  const label = medical ? '正在分析' : '思考';
  return (
    <div className="bg-blue-50/70 border border-blue-100 rounded-xl p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">💭</span>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-blue-600 mb-0.5">{label}</p>
          <p className="text-sm text-slate-700 break-words leading-relaxed">{thought}</p>
        </div>
      </div>
    </div>
  );
}

function ActionCard({ action, actionInput, medical }: { action: string; actionInput: Record<string, unknown> | null; medical?: boolean }) {
  if (!action) return null;
  const label = medical ? (medicalActionMap[action] || '🩺 执行健康分析') : '工具调用';
  return (
    <div className="bg-amber-50/70 border border-amber-100 rounded-xl p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">🔧</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-amber-700 mb-0.5">
            {label}: <span className="font-bold">{action}</span>
          </p>
          {actionInput && Object.keys(actionInput).length > 0 && (
            <pre className="text-xs bg-amber-100/60 rounded-lg p-2 mt-1 overflow-x-auto text-slate-600">
              {JSON.stringify(actionInput, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function ObservationCard({ observation, medical }: { observation: string; medical?: boolean }) {
  if (!observation) return null;
  const label = medical ? '分析结果' : '观察结果';
  return (
    <div className="bg-emerald-50/70 border border-emerald-100 rounded-xl p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">👁️</span>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-emerald-700 mb-0.5">{label}</p>
          <p className="text-xs text-slate-700 font-mono bg-emerald-100/50 p-2 rounded-lg break-all whitespace-pre-wrap">
            {observation.length > 200 ? observation.slice(0, 200) + '...' : observation}
          </p>
        </div>
      </div>
    </div>
  );
}

function ErrorCard({ error }: { error: string }) {
  if (!error) return null;
  return (
    <div className="bg-red-50/70 border border-red-100 rounded-xl p-3">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">⚠️</span>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-red-600 mb-0.5">异常</p>
          <p className="text-sm text-red-700 break-words">{error}</p>
        </div>
      </div>
    </div>
  );
}

interface Props {
  steps: StepInfo[];
  loading: boolean;
  medical?: boolean;
}

export default function AgentSteps({ steps, loading, medical }: Props) {
  if (steps.length === 0 && !loading) return null;

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm font-semibold text-slate-700">
          {medical ? '🫀 分析进度' : 'ReAct 步骤'}
        </span>
        <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">{steps.length} 步</span>
      </div>

      {loading && steps.length === 0 && (
        <div className="text-sm text-slate-400 animate-pulse flex items-center gap-2 py-4">
          <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" />
          <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.1s]" />
          <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.2s]" />
          <span className="ml-1">{medical ? '心康助手 分析中...' : 'Agent 思考中...'}</span>
        </div>
      )}

      <div className="space-y-3">
        {steps.map((step, idx) => (
          <div key={idx} className="relative pl-8 fade-in">
            {/* 时间线连接线 */}
            {idx < steps.length - 1 && (
              <div className="absolute left-[11px] top-7 bottom-[-12px] w-px bg-emerald-200" />
            )}
            {/* 序号圆点 */}
            <div className="absolute left-0 top-0 w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center text-xs font-bold shadow-sm">
              {idx + 1}
            </div>
            <div className="mb-1.5">
              <span className="text-xs font-semibold text-emerald-700">{phaseLabel(step.phase, medical)}</span>
            </div>
            <ThoughtCard thought={step.thought || ''} medical={medical} />
            <ActionCard action={step.action || ''} actionInput={step.action_input} medical={medical} />
            <ObservationCard observation={step.observation || ''} medical={medical} />
            {step.final_answer && !medical && (
              <div className="bg-purple-50/70 border border-purple-100 rounded-xl p-3 mb-2">
                <p className="text-xs font-medium text-purple-600 mb-0.5">✅ 最终答案</p>
                <p className="text-sm text-slate-700">{step.final_answer}</p>
              </div>
            )}
            <ErrorCard error={step.error || ''} />
          </div>
        ))}
      </div>
    </div>
  );
}
