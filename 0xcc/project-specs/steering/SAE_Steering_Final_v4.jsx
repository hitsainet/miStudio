import React, { useState } from 'react';

// Simple Icon components
const Search = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>;
const ChevronDown = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m6 9 6 6 6-6"/></svg>;
const Play = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5,3 19,12 5,21"/></svg>;
const Plus = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>;
const X = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>;
const Copy = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>;
const RefreshCw = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>;
const AlertTriangle = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>;
const Sparkles = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3Z"/></svg>;
const Download = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>;
const Upload = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>;
const Check = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>;
const Database = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>;
const Zap = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>;
const Save = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>;
const Eye = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>;
const ExternalLink = ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>;

export default function SteeringInterface() {
  const [activeMainTab, setActiveMainTab] = useState('Steering');
  const [selectedFeatures, setSelectedFeatures] = useState([
    { id: 7650, name: 'religious_text', strength: 50, layer: 18, color: 'teal' },
    { id: 12331, name: 'scientific_concepts', strength: 30, layer: 18, color: 'blue' }
  ]);
  const [prompt, setPrompt] = useState('In the beginning,');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showFeatureBrowser, setShowFeatureBrowser] = useState(false);
  const [results, setResults] = useState(null);
  const [selectedSAE, setSelectedSAE] = useState('TinyLlama_v1.1_L18_16k');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [genParams, setGenParams] = useState({
    maxTokens: 100,
    temperature: 0.7,
    topP: 0.9,
    numSamples: 2
  });

  const featureColors = ['teal', 'blue', 'purple', 'amber'];

  const mockFeatures = [
    { id: 32, name: 'nuclear_program', freq: '82.0%', interp: '76.7%', topTokens: ['Iran', 'nuclear', 'program'] },
    { id: 7650, name: 'religious_text', freq: '2.3%', interp: '89.2%', topTokens: ['Bible', 'prayer', 'God'] },
    { id: 12331, name: 'scientific_concepts', freq: '4.1%', interp: '85.1%', topTokens: ['research', 'experiment'] },
    { id: 4521, name: 'paris_france', freq: '0.8%', interp: '92.4%', topTokens: ['Paris', 'French', 'Eiffel'] },
    { id: 8892, name: 'code_syntax', freq: '5.2%', interp: '78.3%', topTokens: ['function', 'return'] },
    { id: 3341, name: 'emotional_positive', freq: '3.7%', interp: '81.6%', topTokens: ['happy', 'joy', 'love'] },
  ];

  const saeOptions = [
    { id: 'TinyLlama_v1.1_L18_16k', name: 'TinyLlama_v1.1 - Layer 18 (16k features)', features: 16384 },
    { id: 'TinyLlama_v1.1_L12_16k', name: 'TinyLlama_v1.1 - Layer 12 (16k features)', features: 16384 },
  ];

  const generateResults = () => {
    setIsGenerating(true);
    setTimeout(() => {
      const mockResults = {
        unsteered: {
          samples: [
            "In the beginning, there was nothing but darkness and chaos. The void stretched endlessly across the infinite expanse of what would become existence. Time itself had no meaning.",
            "In the beginning, the world was a simpler place. People lived in small communities and shared what little they had with their neighbors. Life was hard but meaningful.",
          ],
          metrics: { perplexity: 15.2, coherence: 0.95, length: 47 }
        },
        steered: {}
      };

      selectedFeatures.forEach(feature => {
        if (feature.name === 'religious_text') {
          mockResults.steered[feature.id] = {
            feature,
            samples: [
              "In the beginning, God created the heavens and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. The Spirit moved.",
              "In the beginning, the Lord spoke unto the people and said, 'Let there be light,' and there was light, and it was good. The faithful rejoiced in His glory.",
            ],
            metrics: { perplexity: 45.3, coherence: 0.78, behavioral: 0.95, length: 52 }
          };
        } else if (feature.name === 'scientific_concepts') {
          mockResults.steered[feature.id] = {
            feature,
            samples: [
              "In the beginning, the universe formed from a singularity approximately 13.8 billion years ago in an event known as the Big Bang. Matter and energy expanded rapidly.",
              "In the beginning, researchers hypothesized that all fundamental forces were unified in a single superforce at extremely high energies during the Planck epoch.",
            ],
            metrics: { perplexity: 18.7, coherence: 0.91, behavioral: 0.82, length: 58 }
          };
        } else {
          mockResults.steered[feature.id] = {
            feature,
            samples: [
              `In the beginning, [steered output for ${feature.name}] - demonstrates steering with α=${feature.strength}.`,
              `In the beginning, [another sample for ${feature.name}] - showing generation with steering applied.`,
            ],
            metrics: { perplexity: 25.0, coherence: 0.85, behavioral: 0.70, length: 45 }
          };
        }
      });

      setResults(mockResults);
      setIsGenerating(false);
    }, 1500);
  };

  const getStrengthWarning = (strength) => {
    if (strength < -50) return { level: 'danger', message: 'Strong suppression may cause incoherence' };
    if (strength < 0) return { level: 'caution', message: 'Negative values suppress this feature' };
    if (strength > 200) return { level: 'danger', message: 'Very high values may cause gibberish' };
    if (strength > 100) return { level: 'caution', message: 'High strength - monitor coherence' };
    return null;
  };

  const getColorClasses = (color) => {
    const colors = {
      teal: { bg: 'bg-teal-500', border: 'border-teal-500', text: 'text-teal-400', bgLight: 'bg-teal-500/10', borderLight: 'border-teal-500/40' },
      blue: { bg: 'bg-blue-500', border: 'border-blue-500', text: 'text-blue-400', bgLight: 'bg-blue-500/10', borderLight: 'border-blue-500/40' },
      purple: { bg: 'bg-purple-500', border: 'border-purple-500', text: 'text-purple-400', bgLight: 'bg-purple-500/10', borderLight: 'border-purple-500/40' },
      amber: { bg: 'bg-amber-500', border: 'border-amber-500', text: 'text-amber-400', bgLight: 'bg-amber-500/10', borderLight: 'border-amber-500/40' },
    };
    return colors[color] || colors.teal;
  };

  // SAEs Tab
  if (activeMainTab === 'SAEs') {
    return (
      <div className="min-h-screen bg-slate-950 text-gray-100">
        <NavBar activeTab={activeMainTab} setActiveTab={setActiveMainTab} />
        <SAEsTabContent />
      </div>
    );
  }

  // Main Steering Interface
  return (
    <div className="min-h-screen bg-slate-950 text-gray-100">
      <NavBar activeTab={activeMainTab} setActiveTab={setActiveMainTab} />

      <div className="flex h-[calc(100vh-49px)]">
        {/* Left Sidebar - Feature Selection */}
        <div className="w-72 border-r border-slate-800/60 bg-slate-900/50 flex flex-col overflow-hidden">
          {/* SAE Selector */}
          <div className="p-3 border-b border-slate-800/60 shrink-0">
            <label className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 block font-medium">Active SAE</label>
            <div className="relative">
              <select 
                value={selectedSAE}
                onChange={(e) => setSelectedSAE(e.target.value)}
                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-2.5 py-2 text-xs text-white appearance-none cursor-pointer hover:border-teal-500/50 focus:border-teal-500 transition-all"
              >
                {saeOptions.map(sae => (
                  <option key={sae.id} value={sae.id}>{sae.name}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
            </div>
            <div className="flex items-center gap-1.5 mt-1.5 text-[10px]">
              <span className="text-slate-500">16,384 features</span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-500">Layer 18</span>
              <span className="text-slate-600">•</span>
              <span className="text-emerald-400 font-medium">Ready</span>
            </div>
          </div>

          {/* Selected Features */}
          <div className="p-3 border-b border-slate-800/60 shrink-0">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">
                Selected Features ({selectedFeatures.length}/4)
              </span>
              <button 
                onClick={() => setShowFeatureBrowser(true)}
                className="text-[10px] text-teal-400 hover:text-teal-300 flex items-center gap-0.5 font-medium"
              >
                <Plus className="w-3 h-3" /> Add
              </button>
            </div>
            
            <div className="space-y-2">
              {selectedFeatures.map((feature, idx) => {
                const warning = getStrengthWarning(feature.strength);
                const colorClasses = getColorClasses(feature.color);
                return (
                  <div 
                    key={feature.id}
                    className={`bg-slate-800/40 rounded-lg p-2.5 border border-slate-700/30 hover:border-slate-600/50 transition-all group`}
                  >
                    <div className="flex items-start justify-between mb-1.5">
                      <div className="flex items-center gap-1.5">
                        <div className={`w-2 h-2 rounded-full ${colorClasses.bg}`} />
                        <span className="text-xs font-medium text-white">#{feature.id}</span>
                        <span className="text-[10px] text-slate-400">{feature.name}</span>
                      </div>
                      <button 
                        onClick={() => setSelectedFeatures(prev => prev.filter(f => f.id !== feature.id))}
                        className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 transition-all"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                    
                    {/* Strength Slider with Warning Zones */}
                    <div className="mb-1.5">
                      <div className="flex items-center justify-between text-[10px] mb-1">
                        <span className="text-slate-500">Strength (α)</span>
                        <span className={`font-mono font-medium ${
                          warning?.level === 'danger' ? 'text-red-400' :
                          warning?.level === 'caution' ? 'text-amber-400' :
                          colorClasses.text
                        }`}>{feature.strength}</span>
                      </div>
                      
                      <div className="relative h-5">
                        <div className="absolute top-1.5 left-0 right-0 h-2 rounded-full overflow-hidden flex">
                          <div className="w-[12.5%] bg-red-500/30" />
                          <div className="w-[12.5%] bg-amber-500/30" />
                          <div className="w-[25%] bg-slate-700" />
                          <div className="w-[25%] bg-amber-500/30" />
                          <div className="w-[25%] bg-red-500/30" />
                        </div>
                        <input 
                          type="range" 
                          min="-100" 
                          max="300" 
                          value={feature.strength}
                          onChange={(e) => {
                            const newFeatures = [...selectedFeatures];
                            newFeatures[idx].strength = parseInt(e.target.value);
                            setSelectedFeatures(newFeatures);
                          }}
                          className="absolute top-0 left-0 w-full h-5 opacity-0 cursor-pointer z-10"
                        />
                        <div 
                          className={`absolute top-0.5 w-4 h-4 rounded-full border-2 border-white shadow-lg transition-all ${
                            warning?.level === 'danger' ? 'bg-red-500' :
                            warning?.level === 'caution' ? 'bg-amber-500' :
                            colorClasses.bg
                          }`}
                          style={{ left: `calc(${((feature.strength + 100) / 400) * 100}% - 8px)` }}
                        />
                      </div>
                      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                        <span>-100</span>
                        <span>0</span>
                        <span>100</span>
                        <span>200</span>
                        <span>300</span>
                      </div>
                    </div>

                    {warning && (
                      <div className={`flex items-center gap-1 text-[10px] mt-1 p-1.5 rounded ${
                        warning.level === 'danger' ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'
                      }`}>
                        <AlertTriangle className="w-3 h-3 shrink-0" />
                        <span>{warning.message}</span>
                      </div>
                    )}

                    <div className="flex items-center gap-2 mt-1.5">
                      <span className="text-[10px] text-slate-500">Layer:</span>
                      <select 
                        value={feature.layer}
                        onChange={(e) => {
                          const newFeatures = [...selectedFeatures];
                          newFeatures[idx].layer = parseInt(e.target.value);
                          setSelectedFeatures(newFeatures);
                        }}
                        className="bg-slate-900/50 border border-slate-700/50 rounded px-1.5 py-0.5 text-[10px] text-white"
                      >
                        {[...Array(22)].map((_, i) => (
                          <option key={i} value={i}>L{i}</option>
                        ))}
                      </select>
                      <div className="ml-auto flex gap-1">
                        <button className="text-slate-500 hover:text-teal-400 p-0.5"><Eye className="w-3 h-3" /></button>
                        <button className="text-slate-500 hover:text-teal-400 p-0.5"><ExternalLink className="w-3 h-3" /></button>
                      </div>
                    </div>
                  </div>
                );
              })}

              {selectedFeatures.length === 0 && (
                <div className="text-center py-6 text-slate-500">
                  <Zap className="w-6 h-6 mx-auto mb-1.5 opacity-50" />
                  <p className="text-xs">No features selected</p>
                  <p className="text-[10px] mt-0.5 text-slate-600">Click "Add" to browse</p>
                </div>
              )}
            </div>
          </div>

          {/* Quick Presets */}
          <div className="p-3 border-b border-slate-800/60 shrink-0">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 block font-medium">Strength Presets</span>
            <div className="flex gap-1">
              {[
                { label: 'Subtle', value: 10 },
                { label: 'Moderate', value: 50 },
                { label: 'Strong', value: 100 },
              ].map(preset => (
                <button
                  key={preset.label}
                  onClick={() => setSelectedFeatures(prev => prev.map(f => ({ ...f, strength: preset.value })))}
                  className="flex-1 px-1.5 py-1.5 text-[10px] bg-slate-800/50 hover:bg-teal-500/10 border border-slate-700/30 hover:border-teal-500/30 rounded transition-all text-slate-400 hover:text-teal-400"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* Feature Browser */}
          <div className="flex-1 overflow-hidden flex flex-col min-h-0">
            <div className="p-3 pb-2 shrink-0">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 block font-medium">Feature Browser</span>
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                <input 
                  type="text" 
                  placeholder="Search features..."
                  className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg pl-8 pr-2.5 py-1.5 text-xs placeholder-slate-600 focus:border-teal-500 transition-all"
                />
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto px-3 pb-3">
              <div className="space-y-1">
                {mockFeatures.map(feature => {
                  const isSelected = selectedFeatures.find(f => f.id === feature.id);
                  return (
                    <button
                      key={feature.id}
                      onClick={() => {
                        if (selectedFeatures.length < 4 && !isSelected) {
                          const nextColor = featureColors[selectedFeatures.length];
                          setSelectedFeatures(prev => [...prev, { id: feature.id, name: feature.name, strength: 50, layer: 18, color: nextColor }]);
                        }
                      }}
                      disabled={isSelected}
                      className={`w-full text-left p-2 rounded-lg transition-all ${
                        isSelected
                          ? 'bg-teal-500/10 border border-teal-500/30'
                          : 'hover:bg-white/5 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-xs text-white font-medium">#{feature.id}</span>
                        <span className="text-[10px] text-slate-500">{feature.freq}</span>
                      </div>
                      <div className="text-[10px] text-slate-400 mb-1">{feature.name}</div>
                      <div className="flex gap-0.5 flex-wrap">
                        {feature.topTokens.map(token => (
                          <span key={token} className="px-1 py-0.5 bg-slate-800/80 rounded text-[9px] text-slate-500">
                            {token}
                          </span>
                        ))}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Area - Scrollable */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto p-6 space-y-6">
            
            {/* ==================== CONFIGURE SECTION ==================== */}
            <div className="space-y-4">
              {/* Prompt Input */}
              <div className="bg-slate-800/30 rounded-xl border border-slate-700/40 p-5">
                <label className="text-sm font-medium text-white mb-3 block">Prompt</label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt here..."
                  rows={3}
                  className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-4 py-3 text-sm text-white placeholder-slate-600 focus:border-teal-500 focus:ring-1 focus:ring-teal-500/20 transition-all resize-none"
                />
                <div className="flex items-center gap-2 mt-3">
                  <span className="text-[10px] text-slate-500">Quick prompts:</span>
                  {['In the beginning,', 'The capital of France is', 'def fibonacci(n):', 'Once upon a time,'].map(p => (
                    <button
                      key={p}
                      onClick={() => setPrompt(p)}
                      className="px-2 py-1 text-[10px] bg-slate-800/50 border border-slate-700/30 rounded hover:border-teal-500/30 hover:text-teal-400 transition-all text-slate-400"
                    >
                      {p.substring(0, 18)}...
                    </button>
                  ))}
                </div>
              </div>

              {/* Generation Parameters */}
              <div className="bg-slate-800/30 rounded-xl border border-slate-700/40 p-5">
                <div className="flex items-center justify-between mb-4">
                  <label className="text-sm font-medium text-white">Generation Parameters</label>
                  <button 
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="text-xs text-teal-400 hover:text-teal-300 flex items-center gap-1"
                  >
                    {showAdvanced ? 'Hide' : 'Show'} Advanced
                    <ChevronDown className={`w-3 h-3 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                  </button>
                </div>
                
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">Max Tokens</label>
                    <input
                      type="number"
                      value={genParams.maxTokens}
                      onChange={(e) => setGenParams({...genParams, maxTokens: parseInt(e.target.value)})}
                      className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-teal-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">Temperature</label>
                    <input
                      type="number"
                      step="0.1"
                      value={genParams.temperature}
                      onChange={(e) => setGenParams({...genParams, temperature: parseFloat(e.target.value)})}
                      className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-teal-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">Top-p</label>
                    <input
                      type="number"
                      step="0.1"
                      value={genParams.topP}
                      onChange={(e) => setGenParams({...genParams, topP: parseFloat(e.target.value)})}
                      className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-teal-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-1 block">Samples per Config</label>
                    <input
                      type="number"
                      value={genParams.numSamples}
                      onChange={(e) => setGenParams({...genParams, numSamples: parseInt(e.target.value)})}
                      className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-teal-500"
                    />
                  </div>
                </div>

                {showAdvanced && (
                  <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-slate-700/40">
                    <div>
                      <label className="text-xs text-slate-500 mb-1 block">Steering Method</label>
                      <select className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-teal-500">
                        <option>Direct Decoder (Default)</option>
                        <option>SAE-Targeted Steering (SAE-TS)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-slate-500 mb-1 block">Layer Strategy</label>
                      <select className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-teal-500">
                        <option>Single Layer (SAE Layer)</option>
                        <option>All Layers</option>
                        <option>Custom Range</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-slate-500 mb-1 block">Random Seed</label>
                      <input
                        type="number"
                        placeholder="Random"
                        className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-white text-sm placeholder-slate-600 focus:border-teal-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Comparison Configuration Preview */}
              <div className="bg-slate-800/30 rounded-xl border border-slate-700/40 p-5">
                <label className="text-sm font-medium text-white mb-4 block">Comparison Configuration</label>
                
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  {/* Unsteered baseline */}
                  <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/30">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-2 h-2 rounded-full bg-slate-500" />
                      <span className="text-sm font-medium text-white">Unsteered</span>
                    </div>
                    <p className="text-xs text-slate-500">Baseline generation</p>
                  </div>

                  {/* Selected features */}
                  {selectedFeatures.map((feature) => {
                    const colorClasses = getColorClasses(feature.color);
                    const warning = getStrengthWarning(feature.strength);
                    return (
                      <div key={feature.id} className={`bg-slate-900/50 rounded-lg p-3 border ${colorClasses.borderLight}`}>
                        <div className="flex items-center gap-2 mb-2">
                          <div className={`w-2 h-2 rounded-full ${colorClasses.bg}`} />
                          <span className="text-sm font-medium text-white">#{feature.id}</span>
                        </div>
                        <p className="text-xs text-slate-400 mb-1">{feature.name}</p>
                        <p className={`text-xs ${warning ? (warning.level === 'danger' ? 'text-red-400' : 'text-amber-400') : 'text-slate-500'}`}>
                          α = {feature.strength} • L{feature.layer}
                        </p>
                      </div>
                    );
                  })}

                  {/* Add more slot */}
                  {selectedFeatures.length < 3 && (
                    <button 
                      onClick={() => setShowFeatureBrowser(true)}
                      className="bg-slate-900/30 rounded-lg p-3 border border-dashed border-slate-700/50 hover:border-teal-500/50 transition-all group flex flex-col items-center justify-center"
                    >
                      <Plus className="w-5 h-5 text-slate-600 group-hover:text-teal-400 mb-1" />
                      <span className="text-xs text-slate-600 group-hover:text-teal-400">Add Feature</span>
                    </button>
                  )}
                </div>

                {/* Generation Info */}
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-700/40">
                  <div className="text-xs text-slate-500">
                    Will generate <span className="text-white font-medium">{genParams.numSamples}</span> samples × <span className="text-white font-medium">{selectedFeatures.length + 1}</span> configs = <span className="text-teal-400 font-medium">{genParams.numSamples * (selectedFeatures.length + 1)}</span> total outputs
                  </div>
                  <div className="text-xs text-slate-500">
                    Est. time: <span className="text-white">~{Math.ceil((selectedFeatures.length + 1) * genParams.numSamples * 0.8)}s</span>
                  </div>
                </div>
              </div>

              {/* Generate Button */}
              <div className="flex justify-center">
                <button
                  onClick={generateResults}
                  disabled={isGenerating || selectedFeatures.length === 0}
                  className={`flex items-center gap-3 px-8 py-3 rounded-xl font-medium text-white transition-all ${
                    isGenerating || selectedFeatures.length === 0
                      ? 'bg-slate-700 cursor-not-allowed'
                      : 'bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 shadow-lg shadow-teal-500/25'
                  }`}
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Generating Comparison...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Generate Comparison
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* ==================== RESULTS SECTION ==================== */}
            {results && (
              <div className="space-y-6 pt-6 border-t border-slate-700/40">
                {/* Results Header */}
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">Comparison Results</h2>
                  <div className="flex items-center gap-2">
                    <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700/50 rounded-lg hover:border-slate-600 transition-all">
                      <RefreshCw className="w-3.5 h-3.5" /> Regenerate
                    </button>
                    <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700/50 rounded-lg hover:border-slate-600 transition-all">
                      <Save className="w-3.5 h-3.5" /> Save
                    </button>
                    <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700/50 rounded-lg hover:border-slate-600 transition-all">
                      <Download className="w-3.5 h-3.5" /> Export
                    </button>
                  </div>
                </div>

                {/* UNSTEERED SECTION */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-3 h-3 rounded-full bg-slate-500" />
                    <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Unsteered</h3>
                    <span className="text-[10px] text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded">Baseline</span>
                  </div>
                  
                  <div className="bg-slate-800/20 rounded-xl border border-slate-700/30 overflow-hidden">
                    <div className="p-4 space-y-3">
                      {results.unsteered.samples.map((sample, idx) => (
                        <div key={idx} className="bg-slate-900/50 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-slate-500 font-medium">Sample {idx + 1}</span>
                            <button className="text-slate-500 hover:text-white transition-colors"><Copy className="w-3.5 h-3.5" /></button>
                          </div>
                          <p className="text-sm text-slate-300 leading-relaxed">{sample}</p>
                        </div>
                      ))}
                    </div>
                    <div className="px-4 py-3 border-t border-slate-700/30 bg-slate-900/30 flex items-center gap-6">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500">Perplexity:</span>
                        <span className="text-sm font-semibold text-white">{results.unsteered.metrics.perplexity}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500">Coherence:</span>
                        <span className="text-sm font-semibold text-emerald-400">{results.unsteered.metrics.coherence}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500">Avg Length:</span>
                        <span className="text-sm font-semibold text-white">{results.unsteered.metrics.length} tokens</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* STEERED SECTION */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="w-4 h-4 text-teal-400" />
                    <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Steered</h3>
                    <span className="text-[10px] text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded">
                      {Object.keys(results.steered).length} feature{Object.keys(results.steered).length > 1 ? 's' : ''} applied
                    </span>
                  </div>

                  <div className="space-y-4">
                    {Object.entries(results.steered).map(([featureId, data]) => {
                      const colorClasses = getColorClasses(data.feature.color);
                      return (
                        <div 
                          key={featureId} 
                          className={`rounded-xl border-2 ${colorClasses.borderLight} overflow-hidden bg-slate-800/20`}
                        >
                          {/* Feature Header */}
                          <div className={`px-4 py-3 ${colorClasses.bgLight} border-b ${colorClasses.borderLight}`}>
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <div className={`w-3 h-3 rounded-full ${colorClasses.bg}`} />
                                <span className={`text-sm font-semibold ${colorClasses.text}`}>#{data.feature.id}</span>
                                <span className="text-sm text-slate-300 font-medium">{data.feature.name}</span>
                              </div>
                              <div className="flex items-center gap-4">
                                <span className={`text-sm ${colorClasses.text} font-semibold`}>α = {data.feature.strength}</span>
                                <span className="text-xs text-slate-500">Layer {data.feature.layer}</span>
                              </div>
                            </div>
                          </div>

                          {/* Samples */}
                          <div className="p-4 space-y-3">
                            {data.samples.map((sample, idx) => (
                              <div key={idx} className="bg-slate-900/50 rounded-lg p-4">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-xs text-slate-500 font-medium">Sample {idx + 1}</span>
                                  <button className="text-slate-500 hover:text-white transition-colors"><Copy className="w-3.5 h-3.5" /></button>
                                </div>
                                <p className="text-sm text-slate-300 leading-relaxed">{sample}</p>
                              </div>
                            ))}
                          </div>

                          {/* Metrics Footer */}
                          <div className={`px-4 py-3 border-t ${colorClasses.borderLight} bg-slate-900/30 flex items-center gap-6`}>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-500">Perplexity:</span>
                              <span className={`text-sm font-semibold ${data.metrics.perplexity > 30 ? 'text-amber-400' : 'text-white'}`}>
                                {data.metrics.perplexity}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-500">Coherence:</span>
                              <span className="text-sm font-semibold text-white">{data.metrics.coherence}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-500">Behavioral:</span>
                              <span className="text-sm font-semibold text-emerald-400">{data.metrics.behavioral}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-500">Avg Length:</span>
                              <span className="text-sm font-semibold text-white">{data.metrics.length} tokens</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Empty State when no results */}
            {!results && (
              <div className="py-12 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800/30 flex items-center justify-center">
                  <Zap className="w-8 h-8 text-slate-600" />
                </div>
                <p className="text-sm text-slate-400">Configure your steering above and click "Generate Comparison"</p>
                <p className="text-xs text-slate-600 mt-1">Results will appear here</p>
              </div>
            )}

          </div>
        </div>
      </div>

      {/* Feature Browser Modal */}
      {showFeatureBrowser && (
        <FeatureBrowserModal 
          features={mockFeatures}
          selectedFeatures={selectedFeatures}
          featureColors={featureColors}
          onSelect={(feature) => {
            if (selectedFeatures.length < 4 && !selectedFeatures.find(f => f.id === feature.id)) {
              const nextColor = featureColors[selectedFeatures.length];
              setSelectedFeatures(prev => [...prev, { id: feature.id, name: feature.name, strength: 50, layer: 18, color: nextColor }]);
            }
          }}
          onClose={() => setShowFeatureBrowser(false)}
        />
      )}
    </div>
  );
}

// Navigation Bar
function NavBar({ activeTab, setActiveTab }) {
  const tabs = ['Datasets', 'Models', 'Training', 'SAEs', 'Extractions', 'Labeling', 'Steering', 'Templates', 'Monitor'];
  
  return (
    <nav className="border-b border-slate-800/60 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-40">
      <div className="flex items-center justify-between px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="font-semibold text-white text-sm">MechInterp Studio</span>
            <span className="text-[10px] text-slate-500 block leading-tight">Edge AI Feature Discovery</span>
          </div>
        </div>
        
        <div className="flex items-center gap-0.5">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 text-xs rounded-md transition-all ${
                activeTab === tab 
                  ? 'text-teal-400 bg-teal-500/10 font-medium' 
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-slate-500">CPU: <span className="text-slate-300">0%</span></span>
          <span className="text-slate-500">VRAM: <span className="text-emerald-400">0.3/24GB</span></span>
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        </div>
      </div>
    </nav>
  );
}

// SAEs Tab Content
function SAEsTabContent() {
  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white mb-1">SAE Management</h1>
        <p className="text-sm text-slate-400">Download SAEs from HuggingFace or upload your trained SAEs</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Download Section */}
        <div className="bg-slate-800/30 rounded-xl border border-slate-700/40 overflow-hidden">
          <div className="p-4 border-b border-slate-700/40 bg-slate-900/50">
            <div className="flex items-center gap-2">
              <Download className="w-4 h-4 text-teal-400" />
              <h2 className="text-sm font-medium text-white">Download from HuggingFace</h2>
            </div>
          </div>
          <div className="p-4 space-y-4">
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Repository ID</label>
              <input type="text" placeholder="google/gemma-scope-2b-pt-res" className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm placeholder-slate-600 focus:border-teal-500" />
            </div>
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Path (optional)</label>
              <input type="text" placeholder="layer_12/width_16k/canonical" className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm placeholder-slate-600 focus:border-teal-500" />
            </div>
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Access Token (optional)</label>
              <input type="password" placeholder="hf_..." className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm placeholder-slate-600 focus:border-teal-500" />
              <p className="text-[10px] text-slate-600 mt-1">Required for private or gated models</p>
            </div>
            <div className="flex gap-2">
              <button className="flex-1 py-2 text-sm border border-slate-700/50 rounded-lg text-slate-400 hover:text-white hover:border-slate-600">Preview</button>
              <button className="flex-1 py-2 text-sm bg-teal-500 hover:bg-teal-400 rounded-lg text-white font-medium">Download</button>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-slate-800/30 rounded-xl border border-slate-700/40 overflow-hidden">
          <div className="p-4 border-b border-slate-700/40 bg-slate-900/50">
            <div className="flex items-center gap-2">
              <Upload className="w-4 h-4 text-emerald-400" />
              <h2 className="text-sm font-medium text-white">Upload to HuggingFace</h2>
            </div>
          </div>
          <div className="p-4 space-y-4">
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Select Local SAE</label>
              <select className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm focus:border-teal-500">
                <option>TinyLlama_v1.1 - Layer 18 (16k features)</option>
                <option>TinyLlama_v1.1 - Layer 12 (16k features)</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Target Repository</label>
              <input type="text" placeholder="your-username/sae-name" className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm placeholder-slate-600 focus:border-teal-500" />
            </div>
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Access Token (required)</label>
              <input type="password" placeholder="hf_..." className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm placeholder-slate-600 focus:border-teal-500" />
            </div>
            <button className="w-full py-2 text-sm bg-emerald-500 hover:bg-emerald-400 rounded-lg text-white font-medium">Upload SAE</button>
          </div>
        </div>
      </div>

      {/* SAEs List */}
      <div className="mt-6 bg-slate-800/30 rounded-xl border border-slate-700/40 overflow-hidden">
        <div className="p-4 border-b border-slate-700/40 bg-slate-900/50 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-slate-400" />
            <h2 className="text-sm font-medium text-white">Your SAEs</h2>
          </div>
          <div className="flex gap-2 text-[10px]">
            <span className="px-2 py-0.5 bg-teal-500/20 text-teal-400 rounded">Local: 2</span>
            <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">HuggingFace: 0</span>
          </div>
        </div>
        <div className="divide-y divide-slate-700/40">
          {[
            { name: 'TinyLlama_v1.1_L18_16k', model: 'TinyLlama_v1.1', layer: 18, features: 16384, status: 'Ready' },
            { name: 'TinyLlama_v1.1_L12_16k', model: 'TinyLlama_v1.1', layer: 12, features: 16384, status: 'Ready' },
          ].map((sae, idx) => (
            <div key={idx} className="p-4 flex items-center justify-between hover:bg-white/5">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center">
                  <Database className="w-4 h-4 text-slate-500" />
                </div>
                <div>
                  <div className="text-sm font-medium text-white">{sae.name}</div>
                  <div className="text-[10px] text-slate-500">{sae.model} • Layer {sae.layer} • {sae.features.toLocaleString()} features</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="px-2 py-0.5 text-[10px] rounded bg-emerald-500/20 text-emerald-400">{sae.status}</span>
                <button className="text-xs text-teal-400 hover:text-teal-300 px-3 py-1.5 border border-teal-500/30 rounded-lg hover:border-teal-500/50 font-medium">
                  Use in Steering →
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Feature Browser Modal
function FeatureBrowserModal({ features, selectedFeatures, featureColors, onSelect, onClose }) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-xl border border-slate-700/50 w-full max-w-lg max-h-[70vh] overflow-hidden shadow-2xl">
        <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Select Feature to Steer</h2>
          <button onClick={onClose} className="text-slate-500 hover:text-white p-1">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-4 border-b border-slate-700/50">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input type="text" placeholder="Search features by ID or name..." className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg pl-10 pr-3 py-2.5 text-sm placeholder-slate-600 focus:border-teal-500" />
          </div>
        </div>
        <div className="overflow-y-auto max-h-80 p-4 space-y-2">
          {features.map(feature => {
            const isSelected = selectedFeatures.find(f => f.id === feature.id);
            return (
              <button
                key={feature.id}
                onClick={() => { onSelect(feature); onClose(); }}
                disabled={isSelected || selectedFeatures.length >= 4}
                className={`w-full text-left p-4 rounded-lg border transition-all ${
                  isSelected ? 'bg-teal-500/10 border-teal-500/30' :
                  selectedFeatures.length >= 4 ? 'bg-slate-800/30 border-slate-700/30 opacity-50 cursor-not-allowed' :
                  'bg-slate-800/30 border-slate-700/30 hover:border-teal-500/30 hover:bg-teal-500/5'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white">#{feature.id}</span>
                    <span className="text-sm text-slate-400">{feature.name}</span>
                    {isSelected && <span className="flex items-center gap-0.5 px-1.5 py-0.5 bg-teal-500/20 text-teal-400 text-[10px] rounded"><Check className="w-2.5 h-2.5" /> Added</span>}
                  </div>
                  <span className="text-xs text-slate-500">{feature.interp} interp</span>
                </div>
                <div className="flex gap-1 mt-2">
                  {feature.topTokens.map(token => (
                    <span key={token} className="px-2 py-0.5 bg-slate-700/50 rounded text-xs text-slate-400">{token}</span>
                  ))}
                </div>
              </button>
            );
          })}
        </div>
        <div className="p-4 border-t border-slate-700/50 flex justify-between items-center">
          <span className="text-xs text-slate-500">{selectedFeatures.length}/4 features selected</span>
          <button onClick={onClose} className="px-4 py-2 text-sm text-white bg-teal-500 hover:bg-teal-400 rounded-lg font-medium">Done</button>
        </div>
      </div>
    </div>
  );
}
