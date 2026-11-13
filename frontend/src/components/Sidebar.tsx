import { NavLink } from 'react-router-dom';
import SessionList from './SessionList';

interface Props {
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  sessionRefreshKey: number;
}

const links = [
  { to: '/', label: '健康咨询', icon: '🩺' },
  { to: '/monitor', label: '健康趋势', icon: '📈' },
  { to: '/medication', label: '用药管理', icon: '💊' },
  { to: '/documents', label: '健康档案', icon: '📑' },
];

export default function Sidebar({
  activeSessionId,
  onSelectSession,
  onNewSession,
  sessionRefreshKey,
}: Props) {
  return (
    <aside className="w-60 h-screen sidebar-grad text-white flex flex-col shrink-0">
      <div className="px-5 py-5 border-b border-emerald-800/50">
        <h1 className="text-lg font-bold tracking-tight flex items-center gap-2">
          <span className="text-xl">🫀</span> HeartGuard
        </h1>
        <p className="text-xs text-emerald-300/80 mt-0.5">心血管慢病管理 Agent</p>
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
                  ? 'bg-emerald-600/40 text-white font-medium border-r-2 border-emerald-300'
                  : 'text-emerald-100/80 hover:bg-emerald-800/40 hover:text-white'
              }`
            }
          >
            <span className="text-lg">{link.icon}</span>
            {link.label}
          </NavLink>
        ))}
      </nav>

      {/* 咨询记录 */}
      <div className="flex-1 overflow-hidden border-t border-emerald-800/50 dark-scroll">
        <SessionList
          activeSessionId={activeSessionId}
          onSelectSession={onSelectSession}
          onNewSession={onNewSession}
          refreshKey={sessionRefreshKey}
        />
      </div>

      <div className="px-5 py-3 border-t border-emerald-800/50 text-xs text-emerald-400/70">
        HeartGuard Agent v1.0
      </div>
    </aside>
  );
}
