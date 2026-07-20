import { NavLink } from 'react-router-dom';
import SessionList from './SessionList';

interface Props {
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  sessionRefreshKey: number;
}

const links = [
  { to: '/', label: '💬 对话', icon: '💬' },
  { to: '/documents', label: '📄 文档', icon: '📄' },
  { to: '/monitor', label: '📊 监控', icon: '📊' },
];

export default function Sidebar({
  activeSessionId,
  onSelectSession,
  onNewSession,
  sessionRefreshKey,
}: Props) {
  return (
    <aside className="w-56 h-screen bg-slate-900 text-white flex flex-col shrink-0">
      <div className="px-5 py-5 border-b border-slate-700">
        <h1 className="text-lg font-bold tracking-tight">AI Agent</h1>
        <p className="text-xs text-slate-400 mt-0.5">管理控制台</p>
      </div>
      <nav className="py-3">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-slate-700 text-white font-medium'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <span className="text-lg">{link.icon}</span>
            {link.label}
          </NavLink>
        ))}
      </nav>

      {/* 会话列表 */}
      <div className="flex-1 overflow-hidden">
        <SessionList
          activeSessionId={activeSessionId}
          onSelectSession={onSelectSession}
          onNewSession={onNewSession}
          refreshKey={sessionRefreshKey}
        />
      </div>

      <div className="px-5 py-3 border-t border-slate-700 text-xs text-slate-500">
        Enterprise AI Agent v0.1
      </div>
    </aside>
  );
}
