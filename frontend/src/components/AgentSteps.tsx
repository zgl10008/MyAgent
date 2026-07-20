import type { StepInfo } from '../types';

function StepIcon({ phase }: { phase: string }) {
  if (phase === 'react') return <span className="text-lg">🔄</span>;
  return <span className="text-lg">⚙️</span>;
}

function ThoughtCard({ thought }: { thought: string }) {
  if (!thought) return null;
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">💭</span>
        <div>
          <p className="text-xs font-medium text-blue-600 mb-0.5">思考</p>
          <p className="text-sm text-slate-700">{thought}</p>
        </div>
      </div>
    </div>
  );
}

function ActionCard({ action, actionInput }: { action: string; actionInput: Record<string, unknown> | null }) {
  if (!action) return null;
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">🔧</span>
        <div className="flex-1">
          <p className="text-xs font-medium text-amber-600 mb-0.5">
            工具调用: <span className="font-bold">{action}</span>
          </p>
          {actionInput && Object.keys(actionInput).length > 0 && (
            <pre className="text-xs bg-amber-100/50 rounded p-1.5 mt-1 overflow-x-auto text-slate-600">
              {JSON.stringify(actionInput, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function ObservationCard({ observation }: { observation: string }) {
  if (!observation) return null;
  return (
    <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">👁️</span>
        <div>
          <p className="text-xs font-medium text-green-600 mb-0.5">观察结果</p>
          <p className="text-sm text-slate-700 font-mono text-xs bg-green-100/50 p-1.5 rounded">
            {observation.length > 200 ? observation.slice(0, 200) + '...' : observation}
          </p>
        </div>
      </div>
    </div>
  );
}

function FinalAnswerCard({ answer }: { answer: string }) {
  if (!answer) return null;
  return (
    <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">✅</span>
        <div>
          <p className="text-xs font-medium text-purple-600 mb-0.5">最终答案</p>
          <p className="text-sm text-slate-700">{answer}</p>
        </div>
      </div>
    </div>
  );
}

function ErrorCard({ error }: { error: string }) {
  if (!error) return null;
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-3">
      <div className="flex items-start gap-2">
        <span className="text-sm mt-0.5">⚠️</span>
        <div>
          <p className="text-xs font-medium text-red-600 mb-0.5">错误</p>
          <p className="text-sm text-red-700">{error}</p>
        </div>
      </div>
    </div>
  );
}

interface Props {
  steps: StepInfo[];
  loading: boolean;
}

export default function AgentSteps({ steps, loading }: Props) {
  if (steps.length === 0 && !loading) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm font-semibold text-slate-700">ReAct 步骤</span>
        <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">
          {steps.length} 步
        </span>
      </div>

      {loading && steps.length === 0 && (
        <div className="text-sm text-slate-400 animate-pulse flex items-center gap-2 py-4">
          <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
          <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.1s]" />
          <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]" />
          <span className="ml-1">Agent 思考中...</span>
        </div>
      )}

      {steps.map((step, idx) => (
        <div key={idx} className="fade-in">
          <div className="flex items-center gap-2 mb-1.5">
            <StepIcon phase={step.phase} />
            <span className="text-xs font-medium text-slate-500">
              步骤 {idx + 1}
            </span>
          </div>
          <div className="ml-6 pl-3 border-l-2 border-slate-200">
            <ThoughtCard thought={step.thought || ''} />
            <ActionCard action={step.action || ''} actionInput={step.action_input} />
            <ObservationCard observation={step.observation || ''} />
            <FinalAnswerCard answer={step.final_answer || ''} />
            <ErrorCard error={step.error || ''} />
          </div>
        </div>
      ))}
    </div>
  );
}
