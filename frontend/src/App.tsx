import { useEffect } from 'react';
import { DatasetsPanel } from './components/panels/DatasetsPanel';
import { WebSocketProvider, useWebSocketContext } from './contexts/WebSocketContext';
import { useGlobalDatasetProgress } from './hooks/useDatasetProgressV2';
import { setDatasetSubscriptionCallback } from './stores/datasetsStore';

function AppContent() {
  const ws = useWebSocketContext();

  // Set up global dataset progress tracking
  useGlobalDatasetProgress();

  // Wire up the subscription callback so the store can subscribe proactively
  useEffect(() => {
    setDatasetSubscriptionCallback((datasetId: string) => {
      console.log('[App] Proactive subscription callback invoked for dataset:', datasetId);
      ws.subscribe(`datasets/${datasetId}/progress`);
    });
  }, [ws]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-xl font-semibold">MechInterp Studio</h1>
          <p className="text-sm text-slate-400">Edge AI Feature Discovery Platform</p>
        </div>
      </header>
      <main>
        <DatasetsPanel />
      </main>
    </div>
  );
}

function App() {
  return (
    <WebSocketProvider>
      <AppContent />
    </WebSocketProvider>
  );
}

export default App;
