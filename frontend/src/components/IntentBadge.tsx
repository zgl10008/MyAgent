import type { IntentInfo } from '../types';

interface Props {
  intent: IntentInfo | null;
  labelMap?: Record<string, string>;
}

const defaultLabelMap: Record<string, string> = {
  '知识问答': '📖 知识问答', '实时搜索': '🔍 实时查询', '数学计算': '🧮 数值计算',
  '文档管理': '📄 档案管理', '闲聊对话': '💬 日常交流', '问答': '📖 问答', '任务': '🛠️ 任务', '文档': '📄 文档',
};

function getConfidenceColor(conf: number): string {
  if (conf >= 0.7) return 'bg-emerald-100 text-emerald-700 border-emerald-200';
  if (conf >= 0.4) return 'bg-amber-100 text-amber-700 border-amber-200';
  return 'bg-red-100 text-red-700 border-red-200';
}

function getConfidenceLabel(conf: number): string {
  if (conf >= 0.7) return '高'; if (conf >= 0.4) return '中'; return '低';
}

export default function IntentBadge({ intent, labelMap }: Props) {
  if (!intent) return null;
  const map = labelMap || defaultLabelMap;

  return (
    <div className="card p-4">
      <h3 className="text-xs font-semibold text-emerald-600 mb-2 uppercase tracking-wider">🎯 意图识别</h3>
      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
        <span className="text-sm font-medium text-slate-800">
          {map[intent.intent] || intent.intent}
        </span>
        {intent.sub_intent && (
          <>
            <span className="text-slate-300">/</span>
            <span className="text-sm text-slate-600">{intent.sub_intent}</span>
          </>
        )}
        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${getConfidenceColor(intent.confidence)}`}>
          {getConfidenceLabel(intent.confidence)} {(intent.confidence * 100).toFixed(0)}%
        </span>
      </div>
      {intent.rationale && <p className="text-xs text-slate-400 mt-1">{intent.rationale}</p>}
    </div>
  );
}
