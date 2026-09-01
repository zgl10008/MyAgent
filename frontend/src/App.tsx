import { useState, useCallback } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import SessionList from './components/SessionList';
import ChatPage from './pages/ChatPage';
import DocumentsPage from './pages/DocumentsPage';

const tabs = [
  { to: '/', label: '健康咨询', icon: '🩺' },
  { to: '/records', label: '健康档案', icon: '📑' },
];

function App() {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionRefreshKey, setSessionRefreshKey] = useState(0);

  const handleNewSession = useCallback(() => setCurrentSessionId(null), []);
  const handleSelectSession = useCallback((id: string) => setCurrentSessionId(id), []);
  const refreshSessions = useCallback(() => setSessionRefreshKey((k) => k + 1), []);

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-emerald-50/30">
        {/* 左侧导航 */}
        <aside className="w-56 h-screen bg-emerald-900 text-white flex flex-col shrink-0">
          <div className="px-5 py-5 border-b border-emerald-700">
            <h1 className="text-lg font-bold tracking-tight">🫀 Agent/RAG 平台</h1>
            <p className="text-xs text-emerald-300 mt-0.5">心血管慢病默认 Demo</p>
          </div>

          {/* 主导航 Tab */}
          <nav className="py-3">
            {tabs.map((tab) => (
              <NavLink
                key={tab.to}
                to={tab.to}
                end={tab.to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                    isActive
                      ? 'bg-emerald-700 text-white font-medium'
                      : 'text-emerald-200 hover:bg-emerald-800 hover:text-white'
                  }`
                }
              >
                <span className="text-lg">{tab.icon}</span>
                {tab.label}
              </NavLink>
            ))}
          </nav>

          {/* 咨询记录 */}
          <div className="flex-1 overflow-hidden border-t border-emerald-700">
            <div className="px-5 py-2 text-xs text-emerald-400 font-medium uppercase tracking-wider">
              咨询记录
            </div>
            <SessionList
              activeSessionId={currentSessionId}
              onSelectSession={handleSelectSession}
              onNewSession={handleNewSession}
              refreshKey={sessionRefreshKey}
            />
          </div>

          <div className="px-5 py-3 border-t border-emerald-700 text-xs text-emerald-500">
            HeartGuard Agent v1.0
          </div>
        </aside>

        {/* 右侧主内容 */}
        <main className="flex-1 overflow-hidden">
          <Routes>
            <Route
              path="/"
              element={
                <ChatPage
                  sessionId={currentSessionId}
                  onSessionChange={setCurrentSessionId}
                  onSessionsChanged={refreshSessions}
                />
              }
            />
            <Route path="/records" element={<DocumentsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
