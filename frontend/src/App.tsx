import { useEffect, useState } from 'react';
import { DatasetsPanel } from './components/panels/DatasetsPanel';
import { ModelsPanel } from './components/panels/ModelsPanel';
import { TemplatesPanel } from './components/panels/TemplatesPanel';
import { TrainingPanel } from './components/panels/TrainingPanel';
import { ExtractionsPanel } from './components/panels/ExtractionsPanel';
import { LabelingPanel } from './components/panels/LabelingPanel';
import { CircuitsPanel } from './components/panels/CircuitsPanel';
import { JLensPanel } from './components/panels/JLensPanel';
import { ProbeMonitorsPanel } from './components/panels/ProbeMonitorsPanel';
import { SAEsPanel } from './components/panels/SAEsPanel';
import { SteeringPanel } from './components/panels/SteeringPanel';
import { FeatureGroupsPanel } from './components/panels/FeatureGroupsPanel';
import { SystemMonitor } from './components/SystemMonitor/SystemMonitor';
import { SettingsPanel } from './components/panels/SettingsPanel';
import { ApprovalsPanel } from './components/panels/ApprovalsPanel';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { ApprovalsBanner } from './components/layout/ApprovalsBanner';
import { SteeringTaskWatcher } from './components/steering/SteeringTaskWatcher';
import { WebSocketProvider, useWebSocketContext } from './contexts/WebSocketContext';
import { useGlobalDatasetProgress } from './hooks/useDatasetProgress';
import { setDatasetSubscriptionCallback } from './stores/datasetsStore';
import { useUIStore } from './stores/uiStore';
import { useApprovalsLive } from './stores/approvalsStore';
import { isActivePanel, type ActivePanel } from './config/panels';
import { HelpPanel } from './components/help/HelpPanel';
import { useHelpTopic } from './components/help/helpTopicStore';
import { helpHash, parseHelpHash, replaceHash } from './config/helpRoute';

/**
 * The panel to show when the app mounts. A `#/help[/<topic>]` address wins (someone followed a
 * Help link); otherwise the panel saved in localStorage, validated against the panel registry
 * rather than a hand-kept copy of it (the copy is what silently dropped a panel on reload while
 * it worked fine when clicked); otherwise Datasets.
 */
function initialPanel(): ActivePanel {
  const help = parseHelpHash(window.location.hash);
  if (help) {
    useHelpTopic.getState().open(help.topic);
    return 'help';
  }
  const saved = localStorage.getItem('activePanel');
  return isActivePanel(saved) ? saved : 'datasets';
}

function AppContent() {
  // The active panel lives in the UI store (ENH-004, D3) so a Help link anywhere can switch it.
  // It is restored here, once per mount and BEFORE anything below reads the store, so the first
  // render already shows the restored panel.
  useState(() => {
    useUIStore.setState({ activePanel: initialPanel() });
    return true;
  });
  const ws = useWebSocketContext();
  const { sidebar } = useUIStore();
  const activePanel = useUIStore((s) => s.activePanel);
  const setActivePanel = useUIStore((s) => s.setActivePanel);
  const helpTopic = useHelpTopic((s) => s.topic);

  // Theme state management - default to dark mode
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    return saved === 'light' ? 'light' : 'dark';
  });

  // Set up global dataset progress tracking
  useGlobalDatasetProgress();

  // ENH-008 (D7): the ONE approvals poll (15 s) and `mcp/approvals` refresh, for the banner, the
  // menu badge and the Approvals page alike — here, so it keeps running on the Approvals page.
  useApprovalsLive();

  // Save active panel to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('activePanel', activePanel);
  }, [activePanel]);

  // Keep the address in step with Help: on the Help screen it names the topic (so it can be
  // bookmarked); on any other screen a stale `#/help...` is removed. Any other fragment (the
  // J-Lens permalink) is left alone.
  useEffect(() => {
    const onHelp = parseHelpHash(window.location.hash) !== null;
    if (activePanel === 'help') {
      const want = helpHash(helpTopic);
      if (window.location.hash !== want) replaceHash(want);
    } else if (onHelp) {
      replaceHash('');
    }
  }, [activePanel, helpTopic]);

  // A `#/help/<topic>` address typed or followed while the app is open.
  useEffect(() => {
    const onHashChange = () => {
      const help = parseHelpHash(window.location.hash);
      if (!help) return;
      useHelpTopic.getState().open(help.topic);
      useUIStore.getState().setActivePanel('help');
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

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
      <Sidebar
        activePanel={activePanel}
        onPanelChange={(panel) => {
          // The nav entry opens Help at the top, not at whatever topic a link asked for last.
          if (panel === 'help') useHelpTopic.getState().open(null);
          setActivePanel(panel);
        }}
      />

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
        {/* Agent approvals, above EVERY panel (Feature 034, T-97; ENH-008 §2.3): the operator
            must see a waiting request wherever they are. Renders nothing when the queue is
            empty, and nothing on the Approvals panel, which shows the same cards. */}
        <ApprovalsBanner />
        {/* Steering task completions reach the store from here, OUTSIDE the panel switch:
            a batch keeps running while the operator looks at another panel, and its
            prompts used to time out because the subscription lived in SteeringPanel
            (incident 2026-10-28). Renders nothing. */}
        <SteeringTaskWatcher />
        {activePanel === 'datasets' && <DatasetsPanel />}
        {activePanel === 'models' && <ModelsPanel />}
        {activePanel === 'training' && <TrainingPanel />}
        {activePanel === 'extractions' && <ExtractionsPanel />}
        {activePanel === 'labeling' && <LabelingPanel />}
        {activePanel === 'feature-groups' && <FeatureGroupsPanel onNavigateToSteering={() => setActivePanel('steering')} />}
        {activePanel === 'templates' && <TemplatesPanel />}
        {activePanel === 'circuits' && <CircuitsPanel onNavigateToSteering={() => setActivePanel('steering')} />}
        {activePanel === 'probe-monitors' && <ProbeMonitorsPanel />}
        {activePanel === 'jlens' && <JLensPanel />}
        {activePanel === 'saes' && <SAEsPanel onNavigateToSteering={() => setActivePanel('steering')} />}
        {activePanel === 'steering' && <SteeringPanel />}
        {activePanel === 'system' && <SystemMonitor />}
        {activePanel === 'settings' && <SettingsPanel theme={theme} onThemeChange={setTheme} />}
        {activePanel === 'approvals' && <ApprovalsPanel />}
        {activePanel === 'help' && <HelpPanel />}
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
