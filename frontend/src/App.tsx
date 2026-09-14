import { useEffect, useState } from 'react';
import { DatasetsPanel } from './components/panels/DatasetsPanel';
import { ModelsPanel } from './components/panels/ModelsPanel';
import { TemplatesPanel } from './components/panels/TemplatesPanel';
import { TrainingPanel } from './components/panels/TrainingPanel';
import { ExtractionsPanel } from './components/panels/ExtractionsPanel';
import { LabelingPanel } from './components/panels/LabelingPanel';
import { CircuitsPanel } from './components/panels/CircuitsPanel';
import { JLensPanel } from './components/panels/JLensPanel';
import { SAEsPanel } from './components/panels/SAEsPanel';
import { SteeringPanel } from './components/panels/SteeringPanel';
import { FeatureGroupsPanel } from './components/panels/FeatureGroupsPanel';
import { SystemMonitor } from './components/SystemMonitor/SystemMonitor';
import { SettingsPanel } from './components/panels/SettingsPanel';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { WebSocketProvider, useWebSocketContext } from './contexts/WebSocketContext';
import { useGlobalDatasetProgress } from './hooks/useDatasetProgress';
import { setDatasetSubscriptionCallback } from './stores/datasetsStore';
import { useUIStore } from './stores/uiStore';
import { isActivePanel, type ActivePanel } from './config/panels';

function AppContent() {
  const ws = useWebSocketContext();
  const { sidebar } = useUIStore();

  // Restore active panel from localStorage, default to 'datasets'
  const [activePanel, setActivePanel] = useState<ActivePanel>(() => {
    // Validated against the panel registry, not a hand-kept copy of it. The
    // copy is what silently dropped a panel on reload while it worked fine
    // when clicked.
    const saved = localStorage.getItem('activePanel');
    return isActivePanel(saved) ? saved : 'datasets';
  });

  // Theme state management - default to dark mode
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    return saved === 'light' ? 'light' : 'dark';
  });

  // Set up global dataset progress tracking
  useGlobalDatasetProgress();

  // Save active panel to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('activePanel', activePanel);
  }, [activePanel]);

  // Apply theme class to document and persist to localStorage
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Wire up the subscription callback so the store can subscribe proactively
  useEffect(() => {
    setDatasetSubscriptionCallback((datasetId: string) => {
      console.log('[App] Proactive subscription callback invoked for dataset:', datasetId);
      ws.subscribe(`datasets/${datasetId}/progress`);
    });
  }, [ws]);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <Sidebar activePanel={activePanel} onPanelChange={setActivePanel} />

      <Header
        theme={theme}
        onThemeToggle={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        onNavigateToMonitor={() => setActivePanel('system')}
      />

      <main
        className={`
          transition-all duration-300 ease-in-out
          ${sidebar.collapsed ? 'ml-16' : 'ml-56'}
        `}
      >
        {activePanel === 'datasets' && <DatasetsPanel />}
        {activePanel === 'models' && <ModelsPanel />}
        {activePanel === 'training' && <TrainingPanel />}
        {activePanel === 'extractions' && <ExtractionsPanel />}
        {activePanel === 'labeling' && <LabelingPanel />}
        {activePanel === 'feature-groups' && <FeatureGroupsPanel onNavigateToSteering={() => setActivePanel('steering')} />}
        {activePanel === 'templates' && <TemplatesPanel />}
        {activePanel === 'circuits' && <CircuitsPanel onNavigateToSteering={() => setActivePanel('steering')} />}
        {activePanel === 'jlens' && <JLensPanel />}
        {activePanel === 'saes' && <SAEsPanel onNavigateToSteering={() => setActivePanel('steering')} />}
        {activePanel === 'steering' && <SteeringPanel />}
        {activePanel === 'system' && <SystemMonitor />}
        {activePanel === 'settings' && <SettingsPanel />}
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
