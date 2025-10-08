import React from 'react';
import { DatasetsPanel } from './components/panels/DatasetsPanel';

function App() {
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

export default App;
