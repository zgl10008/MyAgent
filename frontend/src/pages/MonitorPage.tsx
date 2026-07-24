import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useState } from 'react';
import RiskBadge, { bloodPressureRisk } from '../components/RiskBadge';

// 模拟 30 天血压数据（实际应从后端记忆系统获取）
const generateBloodPressureData = () => {
  const data = [];
  const now = new Date();
  for (let i = 29; i >= 0; i--) {
    const d = new Date(now); d.setDate(d.getDate() - i);
    const systolic = Math.round(128 + Math.random() * 18);
    const diastolic = Math.round(80 + Math.random() * 10);
    const missed = i === 12; // 模拟有一天忘记服药
    data.push({
      date: `${d.getMonth() + 1}/${d.getDate()}`,
      systolic: missed ? systolic + 22 : systolic,
      diastolic: missed ? diastolic + 12 : diastolic,
      missed,
    });
  }
  return data;
};

const heartRateData = Array.from({ length: 30 }, (_, i) => {
  const d = new Date(); d.setDate(d.getDate() - 29 + i);
  return { date: `${d.getMonth() + 1}/${d.getDate()}`, rate: Math.round(68 + Math.random() * 14) };
});

const bpData = generateBloodPressureData();
const latestSys = bpData[bpData.length - 1].systolic;
const latestDia = bpData[bpData.length - 1].diastolic;
const avgSys = Math.round(bpData.reduce((s, d) => s + d.systolic, 0) / bpData.length);
const avgDia = Math.round(bpData.reduce((s, d) => s + d.diastolic, 0) / bpData.length);

export default function MonitorPage() {
  const [range, setRange] = useState(30);

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-emerald-200 px-6 py-4 bg-white flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-emerald-800">📈 健康趋势</h2>
          <p className="text-xs text-slate-400 mt-0.5">血压 · 心率 · 体重趋势追踪</p>
        </div>
        <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5">
          {[7, 14, 30].map((d) => (
            <button key={d} onClick={() => setRange(d)}
              className={`px-3 py-1 text-xs rounded font-medium transition-colors ${range === d ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'}`}
            >
              {d} 天
            </button>
          ))}
        </div>
      </div>

      <div className="p-6 space-y-6 max-w-4xl">
        {/* 摘要卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-xl border border-emerald-200 p-4">
            <p className="text-xs text-slate-400 mb-1">最新血压</p>
            <p className="text-2xl font-bold text-slate-800">{latestSys}/{latestDia}</p>
            <RiskBadge level={bloodPressureRisk(latestSys, latestDia)} className="mt-1" />
          </div>
          <div className="bg-white rounded-xl border border-emerald-200 p-4">
            <p className="text-xs text-slate-400 mb-1">{range} 天平均血压</p>
            <p className="text-2xl font-bold text-slate-800">{avgSys}/{avgDia}</p>
            <RiskBadge level={bloodPressureRisk(avgSys, avgDia)} className="mt-1" />
          </div>
          <div className="bg-white rounded-xl border border-emerald-200 p-4">
            <p className="text-xs text-slate-400 mb-1">最新心率</p>
            <p className="text-2xl font-bold text-slate-800">{heartRateData[heartRateData.length - 1].rate} <span className="text-sm font-normal text-slate-400">bpm</span></p>
            <RiskBadge level="normal" label="正常" className="mt-1" />
          </div>
        </div>

        {/* 血压趋势图 */}
        <div className="bg-white rounded-xl border border-emerald-200 p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">🩺 血压趋势（近 {range} 天）</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={bpData.slice(-range)} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis domain={[100, 170]} tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #d1d5db', fontSize: 13 }} />
              <ReferenceLine y={140} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: '140 警戒线', position: 'right', fontSize: 10, fill: '#f59e0b' }} />
              <ReferenceLine y={90} stroke="#f59e0b" strokeDasharray="5 5" />
              <Line type="monotone" dataKey="systolic" stroke="#059669" strokeWidth={2} dot={{ r: 3 }} name="收缩压(高压)" />
              <Line type="monotone" dataKey="diastolic" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="舒张压(低压)" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-2 text-xs text-slate-400">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-600 inline-block" /> 收缩压</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 inline-block" /> 舒张压</span>
            <span className="text-amber-500">--- 140/90 警戒线</span>
            <span className="text-xs text-slate-400 ml-auto">🟡 = 疑似漏服药物</span>
          </div>
        </div>

        {/* 心率趋势图 */}
        <div className="bg-white rounded-xl border border-emerald-200 p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">💓 心率趋势（近 {range} 天）</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={heartRateData.slice(-range)} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis domain={[50, 100]} tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #d1d5db', fontSize: 13 }} />
              <Line type="monotone" dataKey="rate" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} name="心率(bpm)" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* AI 健康报告 */}
        <div className="bg-gradient-to-r from-emerald-50 to-blue-50 rounded-xl border border-emerald-200 p-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🤖</span>
            <h3 className="text-sm font-semibold text-emerald-800">Agent 生成：{range} 天健康报告</h3>
          </div>
          <div className="space-y-2 text-sm text-slate-700 leading-relaxed">
            <p>📊 您的血压控制<span className="font-medium text-emerald-600">总体稳定</span>，近 {range} 天平均 <span className="font-bold">{avgSys}/{avgDia}</span> mmHg。</p>
            <p>📈 趋势较上月<span className="font-medium text-emerald-600">改善约 5%</span>，收缩压峰值从 155 降至 146 mmHg。</p>
            <p>⚠️ {bpData.find(d => d.missed)?.date} 有一次异常升高（{bpData.find(d => d.missed)?.systolic}/{bpData.find(d => d.missed)?.diastolic}），当天可能忘记服药。请务必按时服药，设置手机提醒。</p>
            <p>💡 建议：继续保持低盐饮食（每天 &lt;6g），每周至少 150 分钟中等强度运动。</p>
            <p className="text-xs text-slate-400 mt-2">⚠️ 以上为 AI 生成的健康参考，不能替代医生诊断。</p>
          </div>
        </div>
      </div>
    </div>
  );
}
