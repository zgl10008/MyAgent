import type { IntentInfo } from '../types';

interface Props {
  intent: IntentInfo | null;
}

function getConfidenceColor(conf: number): string {
  if (conf >= 0.7) return 'bg-green-100 text-green-700 border-green-200';
  if (conf >= 0.4) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
  return 'bg-red-100 text-red-700 border-red-200';
}

function getConfidenceLabel(conf: number): string {
  if (conf >= 0.7) return '高';
  if (conf >= 0.4) return '中';
  return '低';
}

export default function IntentBadge({ intent }: Props) {
  if (!intent) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <h3 className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
        意图识别
      </h3>
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-sm font-medium text-slate-800">
          {intent.intent}
        </span>
        {intent.sub_intent && (
          <>
            <span className="text-slate-300">/</span>
            <span className="text-sm text-slate-600">{intent.sub_intent}</span>
          </>
        )}
        <span
          className={`text-xs px-2 py-0.5 rounded-full border font-medium ${getConfidenceColor(intent.confidence)}`}
        >
          {getConfidenceLabel(intent.confidence)} {(intent.confidence * 100).toFixed(0)}%
        </span>
      </div>
      {intent.rationale && (
        <p className="text-xs text-slate-400 mt-1">{intent.rationale}</p>
      )}
    </div>
  );
}
