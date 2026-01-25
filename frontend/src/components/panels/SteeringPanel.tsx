/**
 * SteeringPanel - Main panel for feature steering operations.
 *
 * Layout:
 * - Left sidebar: Feature selection (SAE picker, selected features, browser)
 * - Main area: Prompt input, generation config, results
 *
 * Features:
 * - SAE selection with feature browser
 * - Up to 4 features with individual strength controls
 * - Prompt input with generation button
 * - Side-by-side comparison of unsteered vs steered outputs
 * - Save experiments for later reference
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { Play, Loader, AlertCircle, ChevronLeft, ChevronRight, Brain, StopCircle, History, ChevronDown, RotateCcw, Power, PowerOff, Combine } from 'lucide-react';
import { getTaskResult } from '../../api/steering';
import { getSteeringModeStatus, enterSteeringMode, exitSteeringMode } from '../../api/steering';
import { useSteeringStore, selectCanGenerate, selectCanGenerateBatch } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { usePromptTemplatesStore } from '../../stores/promptTemplatesStore';
import { SAEStatus } from '../../types/sae';
import { FeatureSelector } from '../steering/FeatureSelector';
import { GenerationConfig } from '../steering/GenerationConfig';
import { ComparisonPreview } from '../steering/ComparisonPreview';
import { ComparisonResults } from '../steering/ComparisonResults';
import { PromptListEditor } from '../steering/PromptListEditor';
import { COMPONENTS } from '../../config/brand';
import { useSteeringWebSocket } from '../../hooks/useSteeringWebSocket';
import type { SteeringProgressEvent, SteeringCompletedEvent, SteeringFailedEvent } from '../../hooks/useSteeringWebSocket';

export function SteeringPanel() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showRecentDropdown, setShowRecentDropdown] = useState(false);
  const [recoveryTaskId, setRecoveryTaskId] = useState('');
  const [isRecovering, setIsRecovering] = useState(false);
  // Steering mode state
  const [steeringModeActive, setSteeringModeActive] = useState<boolean | null>(null); // null = loading
  const [isTogglingMode, setIsTogglingMode] = useState(false);
  const [modeMessage, setModeMessage] = useState<string | null>(null);

  // Track last WebSocket event time for fallback polling
  const lastWsEventTimeRef = useRef<number>(0);

  const {
    selectedSAE,
    selectedFeatures,
    prompts,
    isGenerating,
    taskId,
    progress,
    progressMessage,
    currentComparison,
    batchState,
    error,
    addPrompt,
    removePrompt,
    updatePrompt,
    clearPrompts,
    replacePromptWithMultiple,
    generateComparison,
    generateBatchComparison,
    abortComparison,
    abortBatch,
    clearBatchResults,
    clearError,
    handleAsyncProgress,
    handleAsyncCompleted,
    handleAsyncFailed,
    recoverActiveTask,
    recoverTaskResult,
    _hasHydrated,
    recentComparisons,
    loadRecentComparison,
    resetSession,
    // Combined mode
    combinedMode,
    setCombinedMode,
    isCombinedGenerating,
    combinedResults,
    generateCombined,
    clearCombinedResults,
  } = useSteeringStore();

  const { saes, fetchSAEs } = useSAEsStore();
  const { templates, fetchTemplates, createTemplate } = usePromptTemplatesStore();

  const canGenerate = selectCanGenerate(useSteeringStore.getState());
  const canGenerateBatch = selectCanGenerateBatch(useSteeringStore.getState());

  // Check if we should use batch mode (more than one non-empty prompt)
  const nonEmptyPrompts = prompts.filter((p) => p.trim().length > 0);
  const isBatchMode = nonEmptyPrompts.length > 1;

  // Load SAEs on mount
  useEffect(() => {
    fetchSAEs();
  }, [fetchSAEs]);

  // Recover active task after page refresh (when state has hydrated)
  useEffect(() => {
    if (_hasHydrated) {
      console.log('[SteeringPanel] State hydrated, checking for active task...');
      recoverActiveTask();
    }
  }, [_hasHydrated, recoverActiveTask]);

  // WebSocket callbacks for async steering task
  const onSteeringProgress = useCallback((data: SteeringProgressEvent) => {
    lastWsEventTimeRef.current = Date.now();
    handleAsyncProgress(data.percent, data.message, data.current_feature, data.current_strength);
  }, [handleAsyncProgress]);

  const onSteeringCompleted = useCallback((data: SteeringCompletedEvent) => {
    lastWsEventTimeRef.current = Date.now();
    handleAsyncCompleted(data.result);
  }, [handleAsyncCompleted]);

  const onSteeringFailed = useCallback((data: SteeringFailedEvent) => {
    lastWsEventTimeRef.current = Date.now();
    handleAsyncFailed(data.error);
  }, [handleAsyncFailed]);

  // Subscribe to WebSocket updates for async steering tasks
  useSteeringWebSocket(taskId, {
    onProgress: onSteeringProgress,
    onCompleted: onSteeringCompleted,
    onFailed: onSteeringFailed,
  });

  // Fallback polling: If no WebSocket events received for 15 seconds while task is active,
  // poll the result endpoint to check if the task completed (WebSocket might have failed)
  useEffect(() => {
    if (!taskId || !isGenerating) {
      return;
    }

    // Reset last event time when a new task starts
    lastWsEventTimeRef.current = Date.now();

    const POLL_INTERVAL_MS = 5000; // Check every 5 seconds
    const WS_TIMEOUT_MS = 15000; // Consider WebSocket dead after 15 seconds of no events

    const pollFallback = async () => {
      const timeSinceLastEvent = Date.now() - lastWsEventTimeRef.current;

      // Only poll if we haven't received a WebSocket event in a while
      if (timeSinceLastEvent < WS_TIMEOUT_MS) {
        return;
      }

      console.log(`[SteeringPanel] Fallback polling: ${timeSinceLastEvent}ms since last WS event`);

      try {
        const result = await getTaskResult(taskId);

        if (result.status.status === 'success' && result.result) {
          console.log('[SteeringPanel] Fallback poll found completed result!');
          handleAsyncCompleted(result.result);
        } else if (result.status.status === 'failure') {
          console.log('[SteeringPanel] Fallback poll found failed task');
          handleAsyncFailed(result.status.error || 'Task failed');
        } else {
          // Task still running - update the last event time to prevent constant polling
          console.log('[SteeringPanel] Fallback poll: task still running');
          // Don't update lastWsEventTimeRef - we want to keep polling if WS is truly dead
        }
      } catch (error) {
        console.warn('[SteeringPanel] Fallback poll error:', error);
      }
    };

    const intervalId = setInterval(pollFallback, POLL_INTERVAL_MS);

    return () => {
      clearInterval(intervalId);
    };
  }, [taskId, isGenerating, handleAsyncCompleted, handleAsyncFailed]);

  const handleGenerate = async () => {
    try {
      if (combinedMode && selectedFeatures.length >= 2) {
        // Combined mode: apply all features together
        await generateCombined(true, true);
      } else if (isBatchMode) {
        await generateBatchComparison(true, true);
      } else {
        await generateComparison(true, true);
      }
    } catch (error) {
      console.error('[SteeringPanel] Generation failed:', error);
    }
  };

  const handleStopBatch = async () => {
    try {
      await abortBatch();
    } catch (error) {
      console.error('[SteeringPanel] Failed to stop batch:', error);
    }
  };

  const handleStopSingle = async () => {
    try {
      await abortComparison();
    } catch (error) {
      console.error('[SteeringPanel] Failed to stop task:', error);
    }
  };

  // Check steering mode status on mount and periodically
  const checkModeStatus = useCallback(async () => {
    try {
      const status = await getSteeringModeStatus();
      setSteeringModeActive(status.active);
    } catch (error) {
      console.error('[SteeringPanel] Failed to check mode status:', error);
      setSteeringModeActive(false);
    }
  }, []);

  useEffect(() => {
    checkModeStatus();
    // Check every 10 seconds
    const interval = setInterval(checkModeStatus, 10000);
    return () => clearInterval(interval);
  }, [checkModeStatus]);

  const handleToggleSteeringMode = async () => {
    setIsTogglingMode(true);
    setModeMessage(null);
    try {
      if (steeringModeActive) {
        // Exit mode
        const result = await exitSteeringMode();
        if (result.success) {
          setSteeringModeActive(false);
          const freedMsg = result.gpu_memory_freed_mb > 0
            ? ` - Freed ${result.gpu_memory_freed_mb}MB GPU`
            : '';
          setModeMessage(`Steering OFF${freedMsg}`);
          // Reset session when exiting
          resetSession();
        } else {
          setModeMessage(result.message || 'Failed to exit');
        }
      } else {
        // Enter mode
        const result = await enterSteeringMode();
        if (result.success) {
          setSteeringModeActive(true);
          setModeMessage('Steering ON');
        } else {
          setModeMessage(result.message || 'Failed to enter');
        }
      }
      // Clear message after 3 seconds
      setTimeout(() => setModeMessage(null), 3000);
    } catch (error) {
      console.error('[SteeringPanel] Toggle mode failed:', error);
      setModeMessage('Toggle failed');
      setTimeout(() => setModeMessage(null), 3000);
    } finally {
      setIsTogglingMode(false);
    }
  };

  const handleSaveExperiment = () => {
    setShowSaveModal(true);
  };

  // Template handlers
  const handleSaveAsTemplate = async (name: string, description: string) => {
    // Filter out empty prompts before saving
    const nonEmptyPromptsList = prompts.filter((p) => p.trim().length > 0);
    await createTemplate({
      name,
      description: description || undefined,
      prompts: nonEmptyPromptsList,
    });
  };

  const handleLoadTemplate = (templatePrompts: string[]) => {
    // Clear current prompts and replace with template prompts
    clearPrompts();
    templatePrompts.forEach((prompt, index) => {
      if (index === 0) {
        updatePrompt(0, prompt);
      } else {
        addPrompt();
        updatePrompt(index, prompt);
      }
    });
  };

  return (
    <div className="h-full flex">
      {/* Sidebar */}
      <div
        className={`transition-all duration-300 ${
          sidebarCollapsed ? 'w-0' : 'w-[420px]'
        } flex-shrink-0 overflow-hidden`}
      >
        <FeatureSelector />
      </div>

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className="flex-shrink-0 w-6 bg-slate-900 border-r border-slate-800 hover:bg-slate-800 flex items-center justify-center"
        title={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
      >
        {sidebarCollapsed ? (
          <ChevronRight className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronLeft className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-xl font-semibold text-slate-100 mb-2">Feature Steering</h1>
              <p className="text-slate-400">
                Steer model outputs by adjusting feature activations during generation
              </p>
            </div>
            {/* Recent comparisons */}
            <div className="flex items-center gap-3">
              {/* Recent Comparisons Dropdown - always visible when not generating */}
              {!isGenerating && (
                <div className="relative">
                  <button
                    onClick={() => setShowRecentDropdown(!showRecentDropdown)}
                    className={`px-3 py-2 flex items-center gap-2 text-sm ${COMPONENTS.button.ghost} ${recentComparisons.length > 0 ? 'text-slate-300 hover:text-slate-100' : 'text-slate-500 hover:text-slate-400'}`}
                  >
                    <History className="w-4 h-4" />
                    Recent {recentComparisons.length > 0 && `(${recentComparisons.length})`}
                    <ChevronDown className={`w-4 h-4 transition-transform ${showRecentDropdown ? 'rotate-180' : ''}`} />
                  </button>
                  {showRecentDropdown && (
                    <>
                      {/* Backdrop to close dropdown */}
                      <div
                        className="fixed inset-0 z-10"
                        onClick={() => setShowRecentDropdown(false)}
                      />
                      {/* Dropdown menu */}
                      <div className="absolute right-0 top-full mt-1 w-80 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-20 overflow-hidden">
                        <div className="px-3 py-2 border-b border-slate-700 text-xs text-slate-400 uppercase tracking-wider">
                          Recent Comparisons
                        </div>
                        <div className="max-h-64 overflow-y-auto">
                          {recentComparisons.length === 0 ? (
                            <div className="px-3 py-4 text-sm text-slate-500">
                              <div className="text-center mb-3">
                                No recent comparisons yet.
                              </div>
                              <div className="border-t border-slate-700 pt-3">
                                <div className="text-xs text-slate-400 mb-2">Recover by Task ID:</div>
                                <div className="flex gap-2">
                                  <input
                                    type="text"
                                    value={recoveryTaskId}
                                    onChange={(e) => setRecoveryTaskId(e.target.value)}
                                    placeholder="Task ID..."
                                    className="flex-1 px-2 py-1 text-xs bg-slate-900 border border-slate-600 rounded text-slate-200 placeholder-slate-500"
                                  />
                                  <button
                                    onClick={async () => {
                                      if (!recoveryTaskId.trim()) return;
                                      setIsRecovering(true);
                                      try {
                                        await recoverTaskResult(recoveryTaskId.trim());
                                        setRecoveryTaskId('');
                                        setShowRecentDropdown(false);
                                      } catch (err) {
                                        // Error is set in store
                                      } finally {
                                        setIsRecovering(false);
                                      }
                                    }}
                                    disabled={isRecovering || !recoveryTaskId.trim()}
                                    className="px-2 py-1 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded disabled:opacity-50"
                                  >
                                    {isRecovering ? '...' : 'Load'}
                                  </button>
                                </div>
                              </div>
                            </div>
                          ) : (
                            recentComparisons.map((recent) => (
                              <button
                                key={recent.id}
                                onClick={() => {
                                  loadRecentComparison(recent.id);
                                  setShowRecentDropdown(false);
                                }}
                                className="w-full px-3 py-2 text-left hover:bg-slate-700/50 transition-colors border-b border-slate-700/50 last:border-b-0"
                              >
                                <div className="text-sm text-slate-200 truncate">
                                  {recent.prompt.length > 50 ? recent.prompt.slice(0, 50) + '...' : recent.prompt}
                                </div>
                                <div className="text-xs text-slate-500 mt-0.5">
                                  {new Date(recent.timestamp).toLocaleString()}
                                </div>
                              </button>
                            ))
                          )}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
              {/* Steering Mode Toggle - Shows ACTION (what clicking will do) */}
              <button
                onClick={handleToggleSteeringMode}
                disabled={isTogglingMode || steeringModeActive === null}
                className={`px-4 py-2 flex items-center gap-2 text-sm font-medium rounded-lg border transition-all ${
                  modeMessage
                    ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-400'
                    : steeringModeActive
                    ? 'border-red-500/50 bg-red-500/10 text-red-400 hover:bg-red-500/20'
                    : 'border-emerald-500/50 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20'
                } disabled:opacity-50`}
                title={steeringModeActive ? 'Stop steering and release GPU memory' : 'Start steering worker on GPU'}
              >
                {isTogglingMode ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : steeringModeActive ? (
                  <PowerOff className="w-4 h-4" />
                ) : (
                  <Power className="w-4 h-4" />
                )}
                {modeMessage || (steeringModeActive === null ? 'Checking...' : steeringModeActive ? 'Stop Steering' : 'Start Steering')}
              </button>
              {/* Reset Session Button - ALWAYS visible to escape stuck state */}
              <button
                onClick={resetSession}
                className={`px-3 py-2 flex items-center gap-2 text-sm ${COMPONENTS.button.ghost} text-slate-400 hover:text-slate-200`}
                title="Reset steering session (clear all state)"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                <span>{error}</span>
              </div>
              <button onClick={clearError} className="text-red-400 hover:text-red-300">
                Dismiss
              </button>
            </div>
          )}

          {/* Steering mode OFF warning */}
          {steeringModeActive === false && (
            <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg flex items-center justify-between">
              <div className="flex items-center gap-3">
                <PowerOff className="w-5 h-5 text-slate-500" />
                <div>
                  <span className="text-slate-300 font-medium">Steering Mode is OFF</span>
                  <span className="text-slate-500 ml-2">— GPU is free. Click "Start Steering" to begin.</span>
                </div>
              </div>
            </div>
          )}

          {/* No SAE selected state */}
          {!selectedSAE && (
            <div className={`${COMPONENTS.card.base} p-8 text-center`}>
              <Brain className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-300 mb-2">
                Select an SAE to begin
              </h3>
              <p className="text-slate-500 max-w-md mx-auto">
                Choose a Sparse Autoencoder from the sidebar to browse its features and start steering.
                {saes.filter((s) => s.status === SAEStatus.READY).length === 0 && (
                  <span className="block mt-2 text-amber-400">
                    No SAEs ready. Go to the SAEs tab to download one.
                  </span>
                )}
              </p>
            </div>
          )}

          {/* Main steering interface */}
          {selectedSAE && (
            <>
              {/* Prompt input - Multi-prompt editor */}
              <div className={`${COMPONENTS.card.base} p-4`}>
                <PromptListEditor
                  prompts={prompts}
                  onAddPrompt={addPrompt}
                  onRemovePrompt={removePrompt}
                  onUpdatePrompt={updatePrompt}
                  onClearPrompts={clearPrompts}
                  onReplacePromptWithMultiple={replacePromptWithMultiple}
                  onSaveAsTemplate={handleSaveAsTemplate}
                  onLoadTemplate={handleLoadTemplate}
                  availableTemplates={templates}
                  onFetchTemplates={fetchTemplates}
                  disabled={isGenerating}
                />
                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center gap-4">
                    <div className="text-sm text-slate-500">
                      {selectedFeatures.length === 0 ? (
                        <span className="text-amber-400">
                          Select at least one feature from the sidebar
                        </span>
                      ) : combinedMode ? (
                        <span className="text-cyan-400">
                          <Combine className="w-3 h-3 inline-block mr-1 -mt-0.5" />
                          {selectedFeatures.length} features combined
                        </span>
                      ) : isBatchMode ? (
                        <span>
                          {selectedFeatures.length} feature{selectedFeatures.length !== 1 ? 's' : ''} × {nonEmptyPrompts.length} prompts
                        </span>
                      ) : (
                        <span>
                          {selectedFeatures.length} feature{selectedFeatures.length !== 1 ? 's' : ''} selected
                        </span>
                      )}
                    </div>
                    {/* Combined Mode toggle - only show with 2+ features */}
                    {selectedFeatures.length >= 2 && !isBatchMode && (
                      <label
                        className={`flex items-center gap-2 text-sm cursor-pointer select-none ${
                          combinedMode ? 'text-cyan-400' : 'text-slate-400 hover:text-slate-300'
                        }`}
                        title="Apply all features simultaneously in a single generation pass"
                      >
                        <input
                          type="checkbox"
                          checked={combinedMode}
                          onChange={(e) => setCombinedMode(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-900"
                          disabled={isGenerating || isCombinedGenerating}
                        />
                        <Combine className="w-4 h-4" />
                        Combined Mode
                      </label>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Stop button for batch mode */}
                    {isGenerating && batchState?.isRunning && (
                      <button
                        onClick={handleStopBatch}
                        className={`px-4 py-2 flex items-center gap-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors`}
                      >
                        <StopCircle className="w-4 h-4" />
                        Stop Batch
                      </button>
                    )}
                    {/* Stop button for single-prompt mode (async task) */}
                    {isGenerating && !batchState && taskId && (
                      <button
                        onClick={handleStopSingle}
                        className={`px-4 py-2 flex items-center gap-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors`}
                      >
                        <StopCircle className="w-4 h-4" />
                        Cancel
                      </button>
                    )}
                    <button
                      onClick={handleGenerate}
                      disabled={(!canGenerate && !canGenerateBatch) || isGenerating || isCombinedGenerating || !steeringModeActive}
                      className={`px-6 py-2 flex items-center gap-2 ${combinedMode ? 'bg-cyan-600 hover:bg-cyan-500 text-white' : COMPONENTS.button.primary} disabled:opacity-50 disabled:cursor-not-allowed`}
                      title={!steeringModeActive ? 'Enter steering mode first' : combinedMode ? 'Apply all features together in one generation' : undefined}
                    >
                      {isGenerating || isCombinedGenerating ? (
                        <>
                          <Loader className="w-4 h-4 animate-spin" />
                          {batchState ? `Processing ${batchState.currentIndex + 1}/${batchState.totalPrompts}...` : isCombinedGenerating ? 'Generating Combined...' : 'Generating...'}
                        </>
                      ) : !steeringModeActive ? (
                        <>
                          <PowerOff className="w-4 h-4" />
                          Start Steering First
                        </>
                      ) : combinedMode ? (
                        <>
                          <Combine className="w-4 h-4" />
                          Generate Combined
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          {isBatchMode ? `Generate ${nonEmptyPrompts.length} Comparisons` : 'Generate Comparison'}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Comparison preview cards */}
              {selectedFeatures.length > 0 && (
                <div className={`${COMPONENTS.card.base} p-4`}>
                  <ComparisonPreview
                    selectedFeatures={selectedFeatures}
                    onAddFeature={
                      selectedFeatures.length < 4
                        ? () => setSidebarCollapsed(false)
                        : undefined
                    }
                  />
                </div>
              )}

              {/* Generation config */}
              <GenerationConfig />

              {/* Progress indicator */}
              {(isGenerating || isCombinedGenerating) && (
                <div className={`${COMPONENTS.card.base} p-4`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-sm ${isCombinedGenerating ? 'text-cyan-400' : 'text-slate-400'}`}>
                      {isCombinedGenerating && <Combine className="w-3 h-3 inline-block mr-1 -mt-0.5" />}
                      {progressMessage || (isCombinedGenerating ? 'Generating with combined features...' : 'Generating...')}
                    </span>
                    <span className={`text-sm font-mono ${isCombinedGenerating ? 'text-cyan-400' : 'text-emerald-400'}`}>
                      {progress.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${isCombinedGenerating ? 'bg-gradient-to-r from-cyan-500 to-cyan-400' : 'bg-gradient-to-r from-emerald-500 to-emerald-400'}`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Results - Single prompt mode */}
              {currentComparison && !isGenerating && !batchState && (
                <ComparisonResults
                  comparison={currentComparison}
                  onSaveExperiment={handleSaveExperiment}
                />
              )}

              {/* Results - Batch mode (show progressively as each prompt completes) */}
              {batchState && batchState.results.some(r => r.status === 'completed' || r.status === 'failed') && (
                <ComparisonResults
                  batchResults={batchState.results}
                  onSaveExperiment={!batchState.isRunning ? handleSaveExperiment : undefined}
                  onClearBatchResults={!batchState.isRunning ? clearBatchResults : undefined}
                  isRunning={batchState.isRunning}
                />
              )}

              {/* Results - Combined mode */}
              {combinedResults && !isCombinedGenerating && (
                <div className={`${COMPONENTS.card.base} p-4`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Combine className="w-5 h-5 text-cyan-400" />
                      <h3 className="text-lg font-medium text-slate-100">Combined Steering Result</h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">
                        {combinedResults.features_applied.length} features × total strength: {combinedResults.total_steering_strength.toFixed(1)}
                      </span>
                      <button
                        onClick={() => clearCombinedResults()}
                        className="px-2 py-1 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded"
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  {/* Features Applied */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {combinedResults.features_applied.map((f, idx) => (
                      <span
                        key={idx}
                        className={`px-2 py-1 text-xs rounded border ${
                          f.strength > 0
                            ? 'border-green-500/30 bg-green-500/10 text-green-400'
                            : f.strength < 0
                            ? 'border-red-500/30 bg-red-500/10 text-red-400'
                            : 'border-slate-500/30 bg-slate-500/10 text-slate-400'
                        }`}
                      >
                        {f.label || `Feature ${f.feature_idx}`}: {f.strength > 0 ? '+' : ''}{f.strength.toFixed(1)}
                      </span>
                    ))}
                  </div>

                  {/* Side-by-side comparison if baseline available */}
                  {combinedResults.baseline_output ? (
                    <div className="grid grid-cols-2 gap-4">
                      {/* Baseline */}
                      <div>
                        <h4 className="text-sm font-medium text-slate-400 mb-2">Baseline (No Steering)</h4>
                        <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700 text-slate-200 text-sm whitespace-pre-wrap">
                          {combinedResults.baseline_output}
                        </div>
                        {combinedResults.baseline_metrics && (
                          <div className="mt-2 text-xs text-slate-500">
                            {combinedResults.baseline_metrics.token_count} tokens • {combinedResults.baseline_metrics.generation_time_ms}ms
                          </div>
                        )}
                      </div>
                      {/* Combined */}
                      <div>
                        <h4 className="text-sm font-medium text-cyan-400 mb-2">
                          <Combine className="w-3 h-3 inline-block mr-1 -mt-0.5" />
                          Combined ({combinedResults.features_applied.length} features)
                        </h4>
                        <div className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/30 text-slate-200 text-sm whitespace-pre-wrap">
                          {combinedResults.combined_output}
                        </div>
                        {combinedResults.combined_metrics && (
                          <div className="mt-2 text-xs text-cyan-400/70">
                            {combinedResults.combined_metrics.token_count} tokens • {combinedResults.combined_metrics.generation_time_ms}ms
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    /* Combined only */
                    <div>
                      <h4 className="text-sm font-medium text-cyan-400 mb-2">
                        <Combine className="w-3 h-3 inline-block mr-1 -mt-0.5" />
                        Combined Output ({combinedResults.features_applied.length} features)
                      </h4>
                      <div className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/30 text-slate-200 text-sm whitespace-pre-wrap">
                        {combinedResults.combined_output}
                      </div>
                      {combinedResults.combined_metrics && (
                        <div className="mt-2 text-xs text-cyan-400/70">
                          {combinedResults.combined_metrics.token_count} tokens • {combinedResults.combined_metrics.generation_time_ms}ms
                        </div>
                      )}
                    </div>
                  )}

                  {/* Timing */}
                  <div className="mt-4 pt-3 border-t border-slate-700 text-xs text-slate-500">
                    Generated in {combinedResults.total_time_ms}ms • {new Date(combinedResults.created_at).toLocaleString()}
                  </div>
                </div>
              )}

              {/* Empty results state */}
              {!currentComparison && !batchState && !combinedResults && !isGenerating && !isCombinedGenerating && selectedFeatures.length > 0 && nonEmptyPrompts.length > 0 && (
                <div className={`${COMPONENTS.card.base} p-8 text-center`}>
                  <Play className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-300 mb-2">
                    Ready to generate
                  </h3>
                  <p className="text-slate-500">
                    Click "Generate Comparison" to see how your selected features affect the output.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Save Experiment Modal */}
      {showSaveModal && (
        <SaveExperimentModal
          onClose={() => setShowSaveModal(false)}
          onSaved={() => setShowSaveModal(false)}
        />
      )}
    </div>
  );
}

// Save Experiment Modal Component
interface SaveExperimentModalProps {
  onClose: () => void;
  onSaved: () => void;
}

function SaveExperimentModal({ onClose, onSaved }: SaveExperimentModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { saveExperiment } = useSteeringStore();

  const handleSave = async () => {
    if (!name.trim()) {
      setError('Name is required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const tagList = tags
        .split(',')
        .map((t) => t.trim())
        .filter((t) => t);
      await saveExperiment(name.trim(), description.trim() || undefined, tagList.length > 0 ? tagList : undefined);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save experiment');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className={`${COMPONENTS.card.base} w-full max-w-md p-6 m-4`}>
        <h3 className="text-lg font-semibold text-slate-100 mb-4">Save Experiment</h3>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My steering experiment"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={3}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Tags <span className="text-slate-500">(comma separated)</span>
            </label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="humor, creative, testing"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500"
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className={`px-4 py-2 ${COMPONENTS.button.ghost}`}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className={`px-4 py-2 ${COMPONENTS.button.primary}`}
            disabled={saving || !name.trim()}
          >
            {saving ? 'Saving...' : 'Save Experiment'}
          </button>
        </div>
      </div>
    </div>
  );
}
