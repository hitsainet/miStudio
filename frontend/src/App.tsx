import { useEffect, useState } from 'react';
import { DatasetsPanel } from './components/panels/DatasetsPanel';
import { ModelsPanel } from './components/panels/ModelsPanel';
import { TemplatesPanel } from './components/panels/TemplatesPanel';
import { TrainingPanel } from './components/panels/TrainingPanel';
import { ExtractionsPanel } from './components/panels/ExtractionsPanel';
import { SystemMonitor } from './components/SystemMonitor/SystemMonitor';
import { CompactGPUStatus } from './components/SystemMonitor/CompactGPUStatus';
import { WebSocketProvider, useWebSocketContext } from './contexts/WebSocketContext';
import { useGlobalDatasetProgress } from './hooks/useDatasetProgressV2';
import { setDatasetSubscriptionCallback } from './stores/datasetsStore';

type ActivePanel = 'datasets' | 'models' | 'training' | 'extractions' | 'templates' | 'system';

function AppContent() {
  const ws = useWebSocketContext();
  // Restore active panel from localStorage, default to 'datasets'
  const [activePanel, setActivePanel] = useState<ActivePanel>(() => {
    const saved = localStorage.getItem('activePanel');
    return (saved === 'models' || saved === 'datasets' || saved === 'training' || saved === 'extractions' || saved === 'templates' || saved === 'system') ? saved : 'datasets';
  });

  // Set up global dataset progress tracking
  useGlobalDatasetProgress();

  // Save active panel to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('activePanel', activePanel);
  }, [activePanel]);

  // Wire up the subscription callback so the store can subscribe proactively
  useEffect(() => {
    setDatasetSubscriptionCallback((datasetId: string) => {
      console.log('[App] Proactive subscription callback invoked for dataset:', datasetId);
      ws.subscribe(`datasets/${datasetId}/progress`);
    });
  }, [ws]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Sticky Header + Navigation Container */}
      <div className="sticky top-0 z-50">
        <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm">
          <div className="max-w-[80%] mx-auto px-6 py-4">
            <h1 className="text-xl font-semibold">MechInterp Studio</h1>
            <p className="text-sm text-slate-400">Edge AI Feature Discovery Platform</p>
          </div>
        </header>

        {/* Navigation Tabs */}
        <nav className="border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm">
        <div className="max-w-[80%] mx-auto px-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex gap-1">
            <button
              onClick={() => setActivePanel('datasets')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activePanel === 'datasets'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Datasets
              {activePanel === 'datasets' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActivePanel('models')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activePanel === 'models'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Models
              {activePanel === 'models' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActivePanel('training')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activePanel === 'training'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Training
              {activePanel === 'training' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActivePanel('extractions')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activePanel === 'extractions'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Extractions
              {activePanel === 'extractions' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActivePanel('templates')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activePanel === 'templates'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Templates
              {activePanel === 'templates' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActivePanel('system')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activePanel === 'system'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Monitor
              {activePanel === 'system' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            </div>

            {/* Compact GPU Status - Right aligned */}
            <CompactGPUStatus onClickMonitor={() => setActivePanel('system')} />
          </div>
        </div>
        </nav>
      </div>

      <main>
        {activePanel === 'datasets' && <DatasetsPanel />}
        {activePanel === 'models' && <ModelsPanel />}
        {activePanel === 'training' && <TrainingPanel />}
        {activePanel === 'extractions' && <ExtractionsPanel />}
        {activePanel === 'templates' && <TemplatesPanel />}
        {activePanel === 'system' && <SystemMonitor />}
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
