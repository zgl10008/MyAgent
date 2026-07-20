import { useState, useEffect } from 'react';
import type { HealthResponse, ReadinessResponse } from '../types';
import { checkHealth, checkReadiness } from '../api/health';

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`w-2.5 h-2.5 rounded-full ${
          ok ? 'bg-green-500' : 'bg-red-500'
        }`}
      />
      <span className="text-sm text-slate-700">{label}</span>
    </div>
  );
}

function StatCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <h3 className="text-sm font-semibold text-slate-500 mb-3">{title}</h3>
      {children}
    </div>
  );
}

export default function MonitorPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [ready, setReady] = useState<ReadinessResponse | null>(null);
  const [error, setError] = useState('');

  const refresh = async () => {
    setError('');
    try {
      const [h, r] = await Promise.all([checkHealth(), checkReadiness()]);
      setHealth(h);
      setReady(r);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-slate-200 px-6 py-3 bg-white">
        <h2 className="text-base font-semibold text-slate-800">📊 系统监控</h2>
        <p className="text-xs text-slate-400 mt-0.5">
          后端服务状态 (每 10 秒自动刷新)
        </p>
      </div>

      <div className="p-6 space-y-4">
        <div className="flex items-center gap-3">
          <button
            onClick={refresh}
            className="px-4 py-2 bg-slate-800 text-white rounded-lg text-sm hover:bg-slate-700 transition-colors"
          >
            立即刷新
          </button>
          {error && <span className="text-sm text-red-500">{error}</span>}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <StatCard title="服务状态">
            {health ? (
              <StatusBadge
                ok={health.status === 'ok'}
                label={`API 服务: ${health.status === 'ok' ? '正常' : health.status}`}
              />
            ) : (
              <div className="text-sm text-slate-400">加载中...</div>
            )}
          </StatCard>

          <StatCard title="运行环境">
            {ready ? (
              <div className="text-sm text-slate-700">
                <p>
                  环境: <span className="font-medium">{ready.app_env}</span>
                </p>
              </div>
            ) : (
              <div className="text-sm text-slate-400">加载中...</div>
            )}
          </StatCard>

          <StatCard title="数据库">
            {ready ? (
              <StatusBadge
                ok={ready.database === 'up'}
                label={`PostgreSQL: ${ready.database === 'up' ? '已连接' : '离线'}`}
              />
            ) : (
              <div className="text-sm text-slate-400">加载中...</div>
            )}
          </StatCard>

          <StatCard title="综合状态">
            {ready ? (
              <StatusBadge
                ok={ready.status === 'ready'}
                label={
                  ready.status === 'ready' ? '全部服务正常' : '部分服务降级'
                }
              />
            ) : (
              <div className="text-sm text-slate-400">加载中...</div>
            )}
          </StatCard>
        </div>
      </div>
    </div>
  );
}
