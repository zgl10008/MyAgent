import { useState, useEffect, useCallback } from 'react';
import { listSessions, deleteSession, type SessionMeta } from '../api/sessions';

interface Props {
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  refreshKey: number; // increment to trigger reload
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 60000) return '刚刚';
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

export default function SessionList({
  activeSessionId,
  onSelectSession,
  onNewSession,
  refreshKey,
}: Props) {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const list = await listSessions();
      setSessions(list);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm('确定删除该会话？')) return;
    try {
      await deleteSession(id);
      await load();
      if (activeSessionId === id) {
        onNewSession();
      }
    } catch {
      alert('删除失败');
    }
  };

  return (
    <div className="border-t border-slate-700 mt-2">
      {/* 折叠标题 */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between px-4 py-2 text-xs text-slate-400 hover:text-slate-300 transition-colors"
      >
        <span>🗂️ 会话列表</span>
        <span className={`transform transition-transform ${collapsed ? '' : 'rotate-90'}`}>
          ▶
        </span>
      </button>

      {/* 新建会话按钮 */}
      {!collapsed && (
        <button
          onClick={onNewSession}
          className="w-full flex items-center gap-2 px-4 py-2 text-xs text-slate-300 hover:bg-slate-800 transition-colors"
        >
          <span className="text-sm">➕</span>
          新建会话
        </button>
      )}

      {/* 会话列表 */}
      {!collapsed && (
        <div className="max-h-[calc(100vh-320px)] overflow-y-auto">
          {loading && sessions.length === 0 && (
            <div className="px-4 py-2 text-xs text-slate-500">加载中...</div>
          )}
          {!loading && sessions.length === 0 && (
            <div className="px-4 py-2 text-xs text-slate-500">暂无会话</div>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => onSelectSession(s.id)}
              className={`group flex items-center gap-2 px-4 py-2 cursor-pointer transition-colors ${
                activeSessionId === s.id
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <span className="text-xs shrink-0">💬</span>
              <div className="flex-1 min-w-0">
                <p className="text-xs truncate">{s.title}</p>
                <p className="text-[10px] text-slate-500">
                  {s.message_count} 条 · {formatTime(s.updated_at)}
                </p>
              </div>
              <button
                onClick={(e) => handleDelete(e, s.id)}
                className="opacity-0 group-hover:opacity-100 text-xs text-slate-500 hover:text-red-400 transition-all shrink-0"
                title="删除"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
