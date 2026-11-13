import { useState, useCallback } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatPage from './pages/ChatPage';
import DocumentsPage from './pages/DocumentsPage';
import MedicationPage from './pages/MedicationPage';
import MonitorPage from './pages/MonitorPage';

function App() {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionRefreshKey, setSessionRefreshKey] = useState(0);

  const handleNewSession = useCallback(() => setCurrentSessionId(null), []);
  const handleSelectSession = useCallback((id: string) => setCurrentSessionId(id), []);
  const refreshSessions = useCallback(() => setSessionRefreshKey((k) => k + 1), []);

  return (
    <BrowserRouter>
      <div className="flex h-screen app-bg">
        <Sidebar
          activeSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          sessionRefreshKey={sessionRefreshKey}
        />
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
            <Route path="/monitor" element={<MonitorPage />} />
            <Route path="/medication" element={<MedicationPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
