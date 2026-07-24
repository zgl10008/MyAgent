import { useState } from 'react';

interface Med {
  name: string; dosage: string; frequency: string; time: string;
  started: string; warning: string; checked: boolean;
}

const defaultMeds: Med[] = [
  { name: '硝苯地平控释片', dosage: '30mg', frequency: '每日1次', time: '早餐后',
    started: '2024-01-15', warning: '避免与葡萄柚同服；可能引起踝部水肿', checked: true },
  { name: '阿托伐他汀钙片', dosage: '20mg', frequency: '每日1次', time: '睡前',
    started: '2024-01-15', warning: '定期检查肝功能；避免大量饮酒', checked: false },
  { name: '阿司匹林肠溶片', dosage: '100mg', frequency: '每日1次', time: '早餐后',
    started: '2024-03-20', warning: '饭后服用减少胃刺激；注意出血风险', checked: true },
];

const history = [
  { date: '2024-03-20', change: '新增阿司匹林 100mg（预防血栓）', reason: '血脂检查提示心血管风险升高' },
  { date: '2024-01-15', change: '硝苯地平 20mg→30mg', reason: '血压控制不达标，调整为标准剂量' },
  { date: '2023-10-01', change: '初始方案：硝苯地平20mg + 阿托伐他汀20mg', reason: '确诊高血压 + 高脂血症' },
];

export default function MedicationPage() {
  const [meds, setMeds] = useState(defaultMeds);
  const toggle = (i: number) => setMeds((prev) => prev.map((m, j) => j === i ? { ...m, checked: !m.checked } : m));
  const done = meds.filter((m) => m.checked).length;

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-emerald-200 px-6 py-4 bg-white">
        <h2 className="text-base font-semibold text-emerald-800">💊 用药管理</h2>
        <p className="text-xs text-slate-400 mt-0.5">当前方案 · 今日打卡 · 用药历史</p>
      </div>

      <div className="p-6 space-y-5 max-w-3xl">
        {/* 今日打卡 */}
        <div className="bg-white rounded-xl border border-emerald-200 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-700">📋 今日用药打卡</h3>
            <span className="text-sm font-bold text-emerald-600">{done}/{meds.length} 已完成</span>
          </div>
          <div className="space-y-3">
            {meds.map((m, i) => (
              <label key={i} className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${m.checked ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-200 hover:bg-slate-100'}`}>
                <input type="checkbox" checked={m.checked} onChange={() => toggle(i)} className="w-4 h-4 accent-emerald-600" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`font-medium text-sm ${m.checked ? 'text-emerald-700 line-through' : 'text-slate-800'}`}>{m.name}</span>
                    <span className="text-xs text-slate-400">{m.dosage}</span>
                  </div>
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="text-xs text-slate-500">{m.time} · {m.frequency}</span>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* 当前方案 */}
        <div className="bg-white rounded-xl border border-emerald-200 p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">📝 当前用药方案</h3>
          <div className="space-y-3">
            {meds.map((m, i) => (
              <div key={i} className="p-3 bg-slate-50 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-sm text-slate-800">{m.name} {m.dosage}</span>
                  <span className="text-xs text-slate-400">|</span>
                  <span className="text-xs text-slate-500">{m.frequency} · {m.time}</span>
                </div>
                <p className="text-xs text-slate-400">开始：{m.started}</p>
                {m.warning && (
                  <p className="text-xs text-amber-600 mt-1 flex items-center gap-1">
                    <span>⚠️</span> {m.warning}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* 用药历史 */}
        <div className="bg-white rounded-xl border border-emerald-200 p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">🕐 方案变更记录</h3>
          <div className="space-y-3">
            {history.map((h, i) => (
              <div key={i} className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className={`w-2.5 h-2.5 rounded-full mt-1.5 ${i === 0 ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                  {i < history.length - 1 && <div className="w-0.5 flex-1 bg-slate-200 mt-1" />}
                </div>
                <div className="pb-3">
                  <p className="text-sm font-medium text-slate-700">{h.change}</p>
                  <p className="text-xs text-slate-400">{h.date} · {h.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
