/**
 * SettingsPanel - Application settings with tabbed interface.
 *
 * Tabs: Endpoints, API Keys, Labeling, Display
 * Data is persisted to the backend database via the settings API.
 * Sensitive values (API keys) are encrypted at rest.
 */

import { useState, useEffect, useRef } from 'react';
import { Plus, Trash2, Eye, EyeOff, Save, AlertCircle, CheckCircle2, RefreshCw, Lock } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useTrainingsStore } from '../../stores/trainingsStore';
import type { CheckpointPrunePreview } from '../../types/training';
import { fetchAPI } from '../../api/client';

type SettingsTab = 'endpoints' | 'api_keys' | 'labeling' | 'storage' | 'display';

const TABS: { id: SettingsTab; label: string }[] = [
  { id: 'endpoints', label: 'Endpoints' },
  { id: 'api_keys', label: 'API Keys' },
  { id: 'labeling', label: 'Labeling' },
  { id: 'storage', label: 'Storage' },
  { id: 'display', label: 'Display' },
];

export function SettingsPanel() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('endpoints');
  const { fetchAll, isLoading } = useSettingsStore();

  useEffect(() => {
    fetchAll();
    // Reset scroll on mount — without this, navigating in from another page
    // (e.g. Features) lands on Settings already scrolled past the tab bar.
    window.scrollTo(0, 0);
  }, [fetchAll]);

  return (
    <div className="px-6 py-8">
      {/* Tabs */}
      <div className="mb-6">
        <div className="border-b border-slate-200 dark:border-slate-800">
          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 font-medium transition-colors relative ${
                  activeTab === tab.id
                    ? 'text-emerald-400'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                }`}
              >
                {tab.label}
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400" />
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      {isLoading && (
        <div className="text-slate-500 text-sm py-4">Loading settings...</div>
      )}
      {activeTab === 'endpoints' && <EndpointsTab />}
      {activeTab === 'api_keys' && <PinGate><ApiKeysTab /></PinGate>}
      {activeTab === 'labeling' && <LabelingTab />}
      {/* PIN-GATED (MIS-E2E-160). The Storage tab arms step-granular
          checkpoint retention — irreversible deletion of training checkpoints —
          and was the one destructive surface in Settings left ungated while
          `api_keys` beside it was protected. `settings-reference.md` described
          the gate as covering "sensitive settings", which a reader would
          reasonably take to include this. */}
      {activeTab === 'storage' && <PinGate><StorageTab /></PinGate>}
      {activeTab === 'display' && <DisplayTab />}
    </div>
  );
}

// ─── PIN Gate ────────────────────────────────────────────────────────────────

type PinStatus = { configured: boolean; bypass_active: boolean };

function PinGate({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<PinStatus | null>(null);
  const [unlocked, setUnlocked] = useState(false);
  const [pin, setPin] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchAPI<PinStatus>('/settings/pin/status')
      .then(setStatus)
      .catch(() => setStatus({ configured: false, bypass_active: false }));
  }, []);

  useEffect(() => {
    if (status && status.configured && !status.bypass_active && !unlocked) {
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [status, unlocked]);

  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    setChecking(true);
    setError(null);
    try {
      const result = await fetchAPI<{ valid: boolean }>('/settings/pin/verify', {
        method: 'POST',
        body: JSON.stringify({ pin }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (result.valid) {
        setUnlocked(true);
      } else {
        setError('Incorrect PIN');
        setPin('');
        inputRef.current?.focus();
      }
    } catch (err) {
      // Log to console so the real failure reason (network, CORS, unexpected JSON)
      // is visible in DevTools even though the user only sees the generic message.
      console.error('[PinGate] PIN verification request failed:', err);
      setError('Failed to verify PIN');
    } finally {
      setChecking(false);
    }
  };

  if (!status) {
    return <div className="text-slate-500 text-sm py-4">Loading...</div>;
  }

  // Bypass active: show warning banner but render content normally
  if (status.bypass_active) {
    return (
      <>
        <div className="mb-6 px-4 py-3 bg-amber-900/30 border border-amber-700/50 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-amber-300">PIN bypass active</p>
            <p className="text-xs text-amber-400/80 mt-0.5">
              <code className="bg-amber-900/40 px-1 rounded">MISTUDIO_BYPASS_PIN=true</code> is set in your
              environment. Reset your PIN below, then remove the flag and restart the backend.
            </p>
          </div>
        </div>
        {children}
      </>
    );
  }

  // No PIN configured, or already unlocked this session
  if (!status.configured || unlocked) {
    return <>{children}</>;
  }

  // Locked — show PIN gate
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-8">
      <div className="text-center">
        <div className="w-14 h-14 rounded-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 flex items-center justify-center mx-auto mb-4">
          <Lock className="w-6 h-6 text-slate-500" />
        </div>
        <h3 className="text-slate-800 dark:text-slate-200 font-medium mb-1">Settings are PIN-protected</h3>
        <p className="text-slate-500 text-sm">Enter your PIN to access settings.</p>
      </div>

      <form onSubmit={handleVerify} className="flex flex-col items-center gap-3 w-64">
        <input
          ref={inputRef}
          type="password"
          value={pin}
          onChange={(e) => { setPin(e.target.value); setError(null); }}
          placeholder="Enter PIN"
          className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-4 py-2.5 text-center text-slate-800 dark:text-slate-200 text-lg tracking-widest focus:border-emerald-500 focus:outline-none placeholder:tracking-normal placeholder:text-sm placeholder:text-slate-400 dark:placeholder:text-slate-600"
        />
        {error && <p className="text-red-400 text-xs">{error}</p>}
        <button
          type="submit"
          disabled={!pin || checking}
          className="w-full px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded transition-colors"
        >
          {checking ? 'Verifying…' : 'Unlock'}
        </button>
      </form>

      {/* Recovery instructions */}
      <div className="max-w-xs text-center border-t border-slate-200 dark:border-slate-800 pt-6">
        <p className="text-xs text-slate-500 font-medium mb-2">Forgot your PIN?</p>
        <p className="text-xs text-slate-600 leading-relaxed">
          Set{' '}
          <code className="text-slate-500 bg-white dark:bg-slate-800 px-1 rounded">MISTUDIO_BYPASS_PIN=true</code>{' '}
          in your server's{' '}
          <code className="text-slate-500 bg-white dark:bg-slate-800 px-1 rounded">.env</code>{' '}
          file and restart the backend. The Settings panel will be accessible without a PIN.
          Reset your PIN in the API Keys tab, then remove the bypass flag and restart again.
        </p>
      </div>
    </div>
  );
}

// ─── PIN Management Section ───────────────────────────────────────────────────

function PinManagementSection() {
  const { remove } = useSettingsStore();
  const [pinStatus, setPinStatus] = useState<PinStatus | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [newPin, setNewPin] = useState('');
  const [confirmPin, setConfirmPin] = useState('');
  const [currentPin, setCurrentPin] = useState('');
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchAPI<PinStatus>('/settings/pin/status').then(setPinStatus).catch(() => {});
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const resetForm = () => {
    setNewPin(''); setConfirmPin(''); setCurrentPin(''); setError(null); setShowForm(false);
  };

  const handleSetPin = async () => {
    if (newPin !== confirmPin) { setError('PINs do not match'); return; }
    if (newPin.length < 4) { setError('PIN must be at least 4 characters'); return; }
    setError(null);
    setSaving(true);
    try {
      await fetchAPI('/settings/pin/set', {
        method: 'POST',
        body: JSON.stringify({
          pin: newPin,
          current_pin: pinStatus?.configured && !pinStatus.bypass_active ? currentPin : undefined,
        }),
        headers: { 'Content-Type': 'application/json' },
      });
      setPinStatus((s) => s ? { ...s, configured: true } : null);
      showToast(pinStatus?.configured ? 'PIN changed' : 'PIN set — settings are now protected');
      resetForm();
    } catch (err: any) {
      // Prefer the backend's detail message (e.g. "Current PIN is incorrect")
      // over the generic Axios/fetch error message string.
      const detail = err?.response?.data?.detail ?? err?.message ?? 'Failed to set PIN';
      setError(detail);
    } finally {
      setSaving(false);
    }
  };

  const handleRemovePin = async () => {
    try {
      await remove('settings_pin_hash');
      setPinStatus((s) => s ? { ...s, configured: false } : null);
      showToast('PIN removed');
    } catch {
      setError('Failed to remove PIN');
    }
  };

  if (!pinStatus) return null;

  return (
    <div className="mt-8">
      <h3 className="text-base font-semibold text-slate-800 dark:text-slate-200 mb-1">Settings Access PIN</h3>
      <p className="text-xs text-slate-500 mb-4">
        {pinStatus.configured
          ? 'A PIN is required to open the Settings panel.'
          : 'Optionally set a PIN to require authentication before accessing Settings.'}
      </p>

      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4">
        {pinStatus.configured && !showForm ? (
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs text-emerald-400 font-medium">
              <Lock className="w-3.5 h-3.5" /> PIN is set
            </span>
            <div className="flex gap-1">
              <button
                onClick={() => setShowForm(true)}
                className="text-xs text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                Change
              </button>
              <button
                onClick={handleRemovePin}
                className="text-xs text-slate-600 dark:text-slate-400 hover:text-red-400 px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {pinStatus.configured && !pinStatus.bypass_active && (
              <div>
                <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Current PIN</label>
                <input
                  type="password"
                  value={currentPin}
                  onChange={(e) => setCurrentPin(e.target.value)}
                  placeholder="Enter current PIN"
                  className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
                />
              </div>
            )}
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">New PIN</label>
              <input
                type="password"
                value={newPin}
                onChange={(e) => { setNewPin(e.target.value); setError(null); }}
                placeholder="Min 4 characters"
                className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Confirm PIN</label>
              <input
                type="password"
                value={confirmPin}
                onChange={(e) => { setConfirmPin(e.target.value); setError(null); }}
                placeholder="Repeat new PIN"
                className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            {error && <p className="text-xs text-red-400">{error}</p>}
            <div className="flex gap-2 pt-1">
              <button
                onClick={handleSetPin}
                disabled={saving || !newPin || !confirmPin}
                className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
              >
                {saving ? 'Saving…' : pinStatus.configured ? 'Change PIN' : 'Set PIN'}
              </button>
              {showForm && (
                <button onClick={resetForm} className="px-3 py-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 text-sm">
                  Cancel
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Endpoints Tab ───────────────────────────────────────────────────────────

function EndpointsTab() {
  const { settings, getByCategory, upsert, remove } = useSettingsStore();
  // MIS-E2E-130(1): filter by the KEY SHAPE, not the category.
  //
  // `category: 'endpoints'` is also carried by `ollama_url` and — worse —
  // `openai_compatible_model`, which is a model NAME. Both rendered in the
  // "Saved Endpoints" list beside real URLs, each with a delete button. So a
  // control labelled "Delete endpoint" removed the labeling model, with no
  // confirmation and nothing on screen explaining what had gone.
  //
  // `handleAdd` writes every real endpoint as `endpoint:<url>`, so that prefix
  // is the actual membership test. The category stays as it is: rewriting it
  // would need a migration for rows already stored, and this is the property
  // the list cares about anyway.
  const endpoints = getByCategory('endpoints').filter((s) => s.key.startsWith('endpoint:'));
  const [url, setUrl] = useState('');
  const [label, setLabel] = useState('');
  const [toast, setToast] = useState<string | null>(null);

  // ── OpenAI-compatible endpoint + model ──────────────────────────────────────
  const compatEndpointSetting = settings.find((s) => s.key === 'openai_compatible_endpoint');
  const compatModelSetting = settings.find((s) => s.key === 'openai_compatible_model');
  const [compatEndpoint, setCompatEndpoint] = useState(compatEndpointSetting?.value ?? '');
  const [compatModel, setCompatModel] = useState(compatModelSetting?.value ?? '');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [fetchModelsError, setFetchModelsError] = useState<string | null>(null);

  useEffect(() => {
    setCompatEndpoint(compatEndpointSetting?.value ?? '');
  }, [compatEndpointSetting?.value]);
  useEffect(() => {
    setCompatModel(compatModelSetting?.value ?? '');
  }, [compatModelSetting?.value]);

  const handleFetchModels = async () => {
    if (!compatEndpoint.trim()) return;
    setFetchingModels(true);
    setFetchModelsError(null);
    setAvailableModels([]);
    try {
      const data = await fetchAPI<{ models: { id: string }[]; total: number }>(
        '/labeling/models/openai',
        { method: 'POST', body: JSON.stringify({ endpoint_url: compatEndpoint.trim() }), headers: { 'Content-Type': 'application/json' } }
      );
      const ids = data.models.map((m) => m.id);
      setAvailableModels(ids);
      if (ids.length > 0 && !compatModel) setCompatModel(ids[0]);
    } catch (err: any) {
      setFetchModelsError(err?.message ?? 'Failed to fetch models');
    } finally {
      setFetchingModels(false);
    }
  };

  const handleSaveCompatSettings = async () => {
    await Promise.all([
      upsert({ key: 'openai_compatible_endpoint', value: compatEndpoint.trim(), is_sensitive: false, category: 'endpoints' }),
      upsert({ key: 'openai_compatible_model', value: compatModel.trim(), is_sensitive: false, category: 'endpoints' }),
    ]);
    setToast('Endpoint & model saved');
    setTimeout(() => setToast(null), 2000);
  };

  // ── Ollama URL setting ──────────────────────────────────────────────────────
  const ollamaUrlSetting = settings.find((s) => s.key === 'ollama_url');
  const [ollamaUrl, setOllamaUrl] = useState(ollamaUrlSetting?.value ?? '');
  useEffect(() => {
    setOllamaUrl(ollamaUrlSetting?.value ?? '');
  }, [ollamaUrlSetting?.value]);

  const handleSaveOllamaUrl = async () => {
    await upsert({ key: 'ollama_url', value: ollamaUrl.trim(), is_sensitive: false, category: 'endpoints' });
    setToast('Ollama URL saved');
    setTimeout(() => setToast(null), 2000);
  };

  const handleAdd = async () => {
    const trimmed = url.trim();
    if (!trimmed) return;
    // Use a sanitized key: endpoint:<url>
    const key = `endpoint:${trimmed}`;
    await upsert({
      key,
      value: JSON.stringify({ url: trimmed, label: label.trim() || undefined, lastUsed: Date.now() }),
      is_sensitive: false,
      category: 'endpoints',
    });
    setUrl('');
    setLabel('');
    setToast('Endpoint saved');
    setTimeout(() => setToast(null), 2000);
  };

  const handleDelete = async (key: string) => {
    await remove(key);
  };

  const parseEndpoint = (value: string) => {
    try {
      return JSON.parse(value) as { url: string; label?: string; lastUsed?: number };
    } catch {
      return { url: value };
    }
  };

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">API Endpoints</h2>
      <p className="text-xs text-slate-500 mb-4">Saved OpenAI-compatible endpoint URLs for labeling jobs.</p>

      {/* Toast */}
      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      {/* OpenAI-Compatible Endpoint + Model */}
      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4 mb-6">
        <label className="block text-sm font-medium text-slate-800 dark:text-slate-200 mb-1">OpenAI-Compatible Endpoint</label>
        <p className="text-xs text-slate-500 mb-3">
          Used by enhanced per-feature labeling and batch labeling jobs.
        </p>
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Endpoint URL</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={compatEndpoint}
                onChange={(e) => { setCompatEndpoint(e.target.value); setAvailableModels([]); setFetchModelsError(null); }}
                placeholder="http://millm-backend.millm.svc.cluster.local:8000/v1"
                className="flex-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none font-mono"
              />
              <button
                onClick={handleFetchModels}
                disabled={fetchingModels || !compatEndpoint.trim()}
                className="flex items-center gap-1.5 px-3 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 text-slate-800 dark:text-slate-200 text-sm rounded transition-colors whitespace-nowrap"
              >
                <RefreshCw className={`w-4 h-4 ${fetchingModels ? 'animate-spin' : ''}`} />
                {fetchingModels ? 'Fetching…' : 'Fetch Models'}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Model</label>
            {availableModels.length > 0 ? (
              <select
                value={compatModel}
                onChange={(e) => setCompatModel(e.target.value)}
                className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
              >
                {availableModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={compatModel}
                onChange={(e) => setCompatModel(e.target.value)}
                placeholder="e.g. gemma-3-27b-it"
                className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
              />
            )}
            {fetchModelsError && (
              <p className="text-xs text-red-400 mt-1">{fetchModelsError}</p>
            )}
            {!fetchModelsError && availableModels.length > 0 && (
              <p className="text-xs text-slate-500 mt-1">{availableModels.length} model(s) available</p>
            )}
            {!fetchModelsError && availableModels.length === 0 && (
              <p className="text-xs text-slate-600 mt-1">Click "Fetch Models" or type a model name manually</p>
            )}
          </div>

          <button
            onClick={handleSaveCompatSettings}
            disabled={!compatEndpoint.trim() || !compatModel.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
          >
            <Save className="w-4 h-4" /> Save
          </button>
        </div>
      </div>

      {/* Ollama / LLM Service URL */}
      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4 mb-6">
        <label className="block text-sm font-medium text-slate-800 dark:text-slate-200 mb-1">Ollama / LLM Service URL</label>
        <p className="text-xs text-slate-500 mb-3">
          Base URL for the local LLM service used for labeling. Overrides the server environment variable.
        </p>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={ollamaUrl}
            onChange={(e) => setOllamaUrl(e.target.value)}
            placeholder="http://llm-host.example.internal:11434"
            className="flex-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none font-mono"
          />
          <button
            onClick={handleSaveOllamaUrl}
            disabled={!ollamaUrl.trim()}
            className="flex items-center gap-1.5 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
          >
            <Save className="w-4 h-4" /> Save
          </button>
          {ollamaUrlSetting && (
            <button
              onClick={async () => {
                // MIS-E2E-130(2): awaited AND caught. This used to clear the
                // field optimistically and drop the promise, so a failed
                // DELETE looked like success and the old value reappeared on
                // reload — with no error shown at any point.
                try {
                  await remove('ollama_url');
                  setOllamaUrl('');
                  setToast('Cleared');
                } catch (err) {
                  setToast(err instanceof Error ? err.message : 'Failed to clear');
                }
                setTimeout(() => setToast(null), 2000);
              }}
              className="p-2 text-slate-500 hover:text-red-400 transition-colors"
              title="Clear (revert to server default)"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
        {ollamaUrlSetting && (
          <p className="text-xs text-slate-600 mt-1">
            Set {new Date(ollamaUrlSetting.updated_at).toLocaleDateString()}
          </p>
        )}
      </div>

      {/* Add form */}
      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4 mb-4">
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">URL</label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://llm-host.example.internal:8001/v1"
              className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Label (optional)</label>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="miLLM GPU Server"
              className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <button
            onClick={handleAdd}
            disabled={!url.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
          >
            <Plus className="w-4 h-4" /> Add Endpoint
          </button>
        </div>
      </div>

      {/* Saved endpoints list */}
      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Saved Endpoints</h3>
      {endpoints.length === 0 ? (
        <p className="text-xs text-slate-500">No saved endpoints yet.</p>
      ) : (
        <div className="space-y-2">
          {endpoints.map((s) => {
            const ep = parseEndpoint(s.value);
            return (
              <div
                key={s.id}
                className="flex items-center justify-between bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg px-4 py-3 group"
              >
                <div>
                  {ep.label && (
                    <div className="text-sm font-medium text-slate-800 dark:text-slate-200">{ep.label}</div>
                  )}
                  <div className="text-xs font-mono text-slate-600 dark:text-slate-400">{ep.url}</div>
                  {ep.lastUsed && (
                    <div className="text-xs text-slate-600 mt-0.5">
                      Last used: {new Date(ep.lastUsed).toLocaleDateString()}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => handleDelete(s.key)}
                  className="p-1.5 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                  title="Delete endpoint"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── API Keys Tab ────────────────────────────────────────────────────────────

const API_KEY_PRESETS = [
  { key: 'openai_api_key', label: 'OpenAI API Key', placeholder: 'sk-proj-...' },
  { key: 'hf_token', label: 'HuggingFace Token', placeholder: 'hf_...' },
];

function ApiKeysTab() {
  const { settings, upsert, remove, fetchAll } = useSettingsStore();
  const apiKeys = settings.filter((s) => s.category === 'api_keys');
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [newValue, setNewValue] = useState('');
  const [showValue, setShowValue] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const handleSave = async (key: string) => {
    if (!newValue.trim()) return;
    await upsert({
      key,
      value: newValue.trim(),
      is_sensitive: true,
      category: 'api_keys',
    });
    setEditingKey(null);
    setNewValue('');
    setShowValue(false);
    setToast(`${key} saved`);
    setTimeout(() => setToast(null), 2000);
    await fetchAll(); // Re-fetch to get masked value
  };

  const handleDelete = async (key: string) => {
    await remove(key);
    setToast(`${key} deleted`);
    setTimeout(() => setToast(null), 2000);
  };

  const getExisting = (key: string) => apiKeys.find((s) => s.key === key);

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">API Keys</h2>
      <p className="text-xs text-slate-500 mb-4">
        Keys are encrypted at rest (AES-256-GCM) and never displayed in full after saving.
      </p>

      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      <div className="space-y-3">
        {API_KEY_PRESETS.map((preset) => {
          const existing = getExisting(preset.key);
          const isEditing = editingKey === preset.key;

          return (
            <div
              key={preset.key}
              className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-slate-800 dark:text-slate-200">{preset.label}</label>
                <div className="flex items-center gap-1">
                  {existing && !isEditing && (
                    <>
                      <button
                        onClick={() => {
                          setEditingKey(preset.key);
                          setNewValue('');
                          setShowValue(false);
                        }}
                        className="text-xs text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete(preset.key)}
                        className="text-xs text-slate-600 dark:text-slate-400 hover:text-red-400 px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                      >
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>

              {isEditing ? (
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <input
                      type={showValue ? 'text' : 'password'}
                      value={newValue}
                      onChange={(e) => setNewValue(e.target.value)}
                      placeholder={preset.placeholder}
                      className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none pr-9"
                      autoFocus
                    />
                    <button
                      onClick={() => setShowValue(!showValue)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      {showValue ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <button
                    onClick={() => handleSave(preset.key)}
                    disabled={!newValue.trim()}
                    className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
                  >
                    <Save className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => { setEditingKey(null); setNewValue(''); setShowValue(false); }}
                    className="px-3 py-2 text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 text-sm"
                  >
                    Cancel
                  </button>
                </div>
              ) : existing ? (
                <div className="flex items-center gap-2">
                  <code className="text-sm font-mono text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-3 py-2 rounded flex-1">
                    {existing.value}
                  </code>
                </div>
              ) : (
                <div>
                  <div className="flex items-center gap-2">
                    <div className="relative flex-1">
                      <input
                        type={showValue && editingKey === null ? 'text' : 'password'}
                        value={editingKey === null && newValue ? newValue : ''}
                        onChange={(e) => { setEditingKey(null); setNewValue(e.target.value); }}
                        onFocus={() => setEditingKey(preset.key)}
                        placeholder={`Enter ${preset.label}...`}
                        className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
                      />
                    </div>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">Not configured</p>
                </div>
              )}

              {existing && !isEditing && existing.updated_at && (
                <p className="text-xs text-slate-600 mt-1">
                  Set {new Date(existing.updated_at).toLocaleDateString()}
                </p>
              )}
            </div>
          );
        })}
      </div>

      <PinManagementSection />
    </div>
  );
}

// ─── Labeling Tab ────────────────────────────────────────────────────────────

function LabelingTab() {
  const { settings, upsert } = useSettingsStore();
  const [toast, setToast] = useState<string | null>(null);

  const getValue = (key: string, defaultVal: string) => {
    const s = settings.find((s) => s.key === key);
    return s?.value ?? defaultVal;
  };

  const handleSave = async (key: string, value: string) => {
    await upsert({ key, value, is_sensitive: false, category: 'labeling' });
    setToast('Saved');
    setTimeout(() => setToast(null), 2000);
  };

  const [batchSize, setBatchSize] = useState(getValue('labeling_default_batch_size', '10'));
  const [maxExamples, setMaxExamples] = useState(getValue('labeling_default_max_examples', '25'));
  const [enhancedWorkers, setEnhancedWorkers] = useState(getValue('enhanced_labeling_max_workers', '8'));
  const [enhancedMethod, setEnhancedMethod] = useState(getValue('enhanced_labeling_method', 'openai_compatible'));
  const [enhancedOpenaiModel, setEnhancedOpenaiModel] = useState(getValue('enhanced_labeling_openai_model', 'gpt-4o-mini'));
  const [openaiModels, setOpenaiModels] = useState<string[]>([]);
  const [fetchingOpenaiModels, setFetchingOpenaiModels] = useState(false);
  const [openaiModelsError, setOpenaiModelsError] = useState<string | null>(null);

  // Has an OpenAI API key been configured on the API Keys tab?
  const hasOpenaiApiKey = settings.some((s) => s.key === 'openai_api_key' && s.value);

  const handleFetchOpenaiModels = async () => {
    setFetchingOpenaiModels(true);
    setOpenaiModelsError(null);
    try {
      const data = await fetchAPI<{ models: { id: string }[]; total: number }>(
        '/labeling/models/openai',
        {
          method: 'POST',
          body: JSON.stringify({ endpoint_url: 'https://api.openai.com/v1' }),
        }
      );
      const ids = (data.models || []).map((m) => m.id).filter(Boolean);
      if (ids.length === 0) {
        setOpenaiModelsError('No models returned from OpenAI API.');
      } else {
        setOpenaiModels(ids);
        if (!ids.includes(enhancedOpenaiModel)) {
          setEnhancedOpenaiModel(ids.find((id) => id === 'gpt-4o-mini') || ids[0]);
        }
      }
    } catch (err) {
      setOpenaiModelsError(err instanceof Error ? err.message : 'Failed to fetch models');
    } finally {
      setFetchingOpenaiModels(false);
    }
  };

  // Sync local state when settings load
  // MIS-E2E-130(3): seed the fields ONCE, not on every `settings` change.
  //
  // This effect depended on `[settings]`, and `upsert` refreshes that array.
  // So saving any one card rewrote all five inputs from the server — silently
  // discarding edits the user had typed into the others but not yet saved.
  // The trigger was invisible: you save card A and card B reverts.
  //
  // The fields are *pre-fill* values, so seeding them when the settings first
  // arrive is the whole intent. `seededRef` makes that explicit rather than
  // relying on a dependency array to approximate it.
  const seededRef = useRef(false);
  useEffect(() => {
    if (seededRef.current || settings.length === 0) return;
    seededRef.current = true;
    setBatchSize(getValue('labeling_default_batch_size', '10'));
    setMaxExamples(getValue('labeling_default_max_examples', '25'));
    setEnhancedWorkers(getValue('enhanced_labeling_max_workers', '8'));
    setEnhancedMethod(getValue('enhanced_labeling_method', 'openai_compatible'));
    setEnhancedOpenaiModel(getValue('enhanced_labeling_openai_model', 'gpt-4o-mini'));
  }, [settings, getValue]);

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">Labeling Defaults</h2>
      <p className="text-xs text-slate-500 mb-4">
        These values pre-fill when starting a new labeling job.
      </p>

      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4 space-y-4">
        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Default Batch Size</label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(e.target.value)}
            min={1}
            max={100}
            className="w-32 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Default Max Examples per Feature</label>
          <input
            type="number"
            value={maxExamples}
            onChange={(e) => setMaxExamples(e.target.value)}
            min={5}
            max={100}
            className="w-32 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <button
          onClick={async () => {
            await handleSave('labeling_default_batch_size', batchSize);
            await handleSave('labeling_default_max_examples', maxExamples);
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded transition-colors"
        >
          <Save className="w-4 h-4" /> Save Defaults
        </button>
      </div>

      {/* Enhanced labeling settings */}
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mt-8 mb-1">Enhanced Labeling</h2>
      <p className="text-xs text-slate-500 mb-4">
        Settings for the two-pass per-feature labeling triggered from the Feature Detail modal.
      </p>
      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4 space-y-4">
        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Labeling Method</label>
          <select
            value={enhancedMethod}
            onChange={(e) => setEnhancedMethod(e.target.value)}
            className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
          >
            <option value="openai">OpenAI (requires api-key)</option>
            <option value="openai_compatible">OpenAI-Compatible (miLLM, Ollama, vLLM, etc.)</option>
          </select>
        </div>

        {enhancedMethod === 'openai' && (
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">OpenAI Model</label>
            <div className="flex gap-2">
              {openaiModels.length > 0 ? (
                <select
                  value={enhancedOpenaiModel}
                  onChange={(e) => setEnhancedOpenaiModel(e.target.value)}
                  className="flex-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
                >
                  {openaiModels.map((m) => (
                    <option key={m} value={m}>{m}{m === 'gpt-4o-mini' ? ' (recommended)' : ''}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={enhancedOpenaiModel}
                  onChange={(e) => setEnhancedOpenaiModel(e.target.value)}
                  placeholder="e.g. gpt-4o-mini"
                  className="flex-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
                />
              )}
              <button
                onClick={handleFetchOpenaiModels}
                disabled={fetchingOpenaiModels || !hasOpenaiApiKey}
                title={!hasOpenaiApiKey ? 'Set the OpenAI API key first' : 'Fetch available models from OpenAI'}
                className="flex items-center gap-1.5 px-3 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-slate-800 dark:text-slate-200 text-sm rounded transition-colors whitespace-nowrap"
              >
                <RefreshCw className={`w-4 h-4 ${fetchingOpenaiModels ? 'animate-spin' : ''}`} />
                {fetchingOpenaiModels ? 'Fetching…' : 'Fetch Models'}
              </button>
            </div>
            {openaiModelsError && (
              <p className="text-xs text-red-400 mt-1">{openaiModelsError}</p>
            )}
            {!openaiModelsError && openaiModels.length > 0 && (
              <p className="text-xs text-slate-500 mt-1">{openaiModels.length} model(s) available from OpenAI</p>
            )}
            {!openaiModelsError && openaiModels.length === 0 && (
              hasOpenaiApiKey ? (
                <p className="text-xs text-emerald-500 mt-1">
                  ✓ OpenAI API key is configured on the API Keys tab — click "Fetch Models" to list available models
                </p>
              ) : (
                <p className="text-xs text-amber-400 mt-1">
                  ⚠ Set the OpenAI API key on the <strong>API Keys</strong> tab before starting a job.
                </p>
              )
            )}
          </div>
        )}

        {enhancedMethod === 'openai_compatible' && (
          <p className="text-xs text-slate-500 -mt-2">
            Uses the endpoint and model configured on the <strong>Endpoints</strong> tab.
          </p>
        )}

        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Max Parallel Workers (Pass 1)</label>
          <input
            type="number"
            value={enhancedWorkers}
            onChange={(e) => setEnhancedWorkers(e.target.value)}
            min={1}
            max={20}
            className="w-32 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-none"
          />
          <p className="text-xs text-slate-500 mt-1">
            Concurrent LLM calls during per-example summarization. Reduce if the
            inference server returns 500 errors (recommended: 4–8 for a single GPU).
          </p>
        </div>
        <button
          onClick={async () => {
            await handleSave('enhanced_labeling_method', enhancedMethod);
            if (enhancedMethod === 'openai') {
              await handleSave('enhanced_labeling_openai_model', enhancedOpenaiModel);
            }
            await handleSave('enhanced_labeling_max_workers', enhancedWorkers);
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded transition-colors"
        >
          <Save className="w-4 h-4" /> Save
        </button>
      </div>
    </div>
  );
}

// ─── Display Tab ─────────────────────────────────────────────────────────────

function DisplayTab() {
  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">Display Preferences</h2>
      <p className="text-xs text-slate-500 mb-4">
        Saved locally in your browser. These settings don't sync across devices.
      </p>

      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-slate-500" />
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Display preferences are managed via the sidebar collapse toggle and theme button in the header.
            Additional display settings will be added here in a future update.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Storage Tab ─────────────────────────────────────────────────────────────
/**
 * Checkpoint retention settings.
 *
 * Pruning permanently deletes checkpoint files, so the shipped defaults are
 * deliberately inert: disabled, and dry-run when first enabled. An operator
 * reviews the dry-run report before any deletion happens.
 */
export function StorageTab() {
  const { settings, upsert } = useSettingsStore();
  const [toast, setToast] = useState<string | null>(null);

  const getValue = (key: string, defaultVal: string) => {
    const s = settings.find((s) => s.key === key);
    return s?.value ?? defaultVal;
  };

  // Deliberately does NOT toast: the caller toasts once, after ALL writes
  // succeed, so a partial failure can never render as "Saved".
  const handleSave = async (key: string, value: string) => {
    await upsert({ key, value, is_sensitive: false, category: 'general' });
  };

  const [saveError, setSaveError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState(getValue('checkpoint_prune_enabled', 'false'));
  const [dryRun, setDryRun] = useState(getValue('checkpoint_prune_dry_run', 'true'));
  const [keepLast, setKeepLast] = useState(getValue('checkpoint_prune_keep_last', '2'));
  const [keepBest, setKeepBest] = useState(getValue('checkpoint_prune_keep_best', 'true'));
  const [minAgeHours, setMinAgeHours] = useState(
    getValue('checkpoint_prune_min_age_hours', '24')
  );

  // Re-sync once the settings request resolves.
  useEffect(() => {
    setEnabled(getValue('checkpoint_prune_enabled', 'false'));
    setDryRun(getValue('checkpoint_prune_dry_run', 'true'));
    setKeepLast(getValue('checkpoint_prune_keep_last', '2'));
    setKeepBest(getValue('checkpoint_prune_keep_best', 'true'));
    setMinAgeHours(getValue('checkpoint_prune_min_age_hours', '24'));
  }, [settings]);

  const isTruthy = (v: string) => ['1', 'true', 'yes', 'on'].includes(v.trim().toLowerCase());

  // "Run cleanup now" across EVERY training.
  //
  // The sweep was previously reachable only from the daily scheduler, and that
  // scheduler no-ops while `checkpoint_prune_enabled` is false — the shipped
  // default. So 84 GB of prunable checkpoints accumulated with no way to
  // reclaim them from the UI except previewing and pruning one training at a
  // time (2026-08-28).
  const [sweepBusy, setSweepBusy] = useState(false);
  const [sweepPreview, setSweepPreview] = useState<null | {
    trainings_affected: number;
    total_checkpoints: number;
    estimated_bytes: number;
    policy: { dry_run: boolean };
  }>(null);
  const [sweepMsg, setSweepMsg] = useState<string | null>(null);
  const [sweepError, setSweepError] = useState<string | null>(null);

  const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

  const loadSweepPreview = async () => {
    setSweepBusy(true);
    setSweepError(null);
    setSweepMsg(null);
    try {
      const res = await fetchAPI<{ data: any }>(
        '/api/v1/trainings/checkpoints/prune-preview-all'
      );
      setSweepPreview(res.data);
    } catch (e: any) {
      setSweepError(e?.message || 'Failed to preview the sweep');
    } finally {
      setSweepBusy(false);
    }
  };

  const runSweep = async () => {
    setSweepBusy(true);
    setSweepError(null);
    setSweepMsg(null);
    try {
      // MIS-E2E-128 applies here too. The Celery task re-reads
      // `checkpoint_prune_dry_run` from settings when it EXECUTES, so the local
      // form state — which may be edited and unsaved — cannot be trusted to
      // describe what is about to happen. Confirm against the policy the server
      // will actually use, re-read now.
      const live = await fetchAPI<{ data: any }>(
        '/api/v1/trainings/checkpoints/prune-preview-all'
      );
      const willDelete = live.data?.policy?.dry_run === false;
      const count = live.data?.total_checkpoints ?? 0;
      const bytes = live.data?.estimated_bytes ?? 0;

      const message = willDelete
        ? `PERMANENTLY DELETE ${count} checkpoint file(s), reclaiming ${gb(bytes)}.\n\nThis cannot be undone. Continue?`
        : `Dry run: this will REPORT on ${count} checkpoint file(s) (${gb(bytes)}) without deleting anything.\n\nContinue?`;

      if (!window.confirm(message)) {
        setSweepBusy(false);
        return;
      }

      await fetchAPI('/api/v1/trainings/checkpoints/prune-all', { method: 'POST' });
      setSweepMsg(
        willDelete
          ? `Sweep queued — removing ${count} checkpoint(s) in the background.`
          : 'Sweep queued in DRY RUN — it will report without deleting.'
      );
      setSweepPreview(null);
    } catch (e: any) {
      // Fail CLOSED: if the live policy could not be read, do not run. Assuming
      // dry-run here is how a destructive sweep gets launched behind a dialog
      // promising a report.
      setSweepError(
        e?.message ||
          'Could not confirm the live retention policy — sweep not started'
      );
    } finally {
      setSweepBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-slate-900 dark:text-white font-medium mb-1">
          Checkpoint Retention
        </h3>
        <p className="text-xs text-slate-500">
          Training saves a checkpoint every N steps and never removes them. Pruning
          reclaims that space. Deletions are permanent — the best checkpoint, the
          most recent steps, and any active training are always protected.
        </p>
      </div>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={isTruthy(enabled)}
          onChange={(e) => setEnabled(e.target.checked ? 'true' : 'false')}
          className="rounded"
        />
        <span className="text-sm text-slate-700 dark:text-slate-300">
          Enable scheduled pruning (runs daily)
        </span>
      </label>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={isTruthy(dryRun)}
          onChange={(e) => setDryRun(e.target.checked ? 'true' : 'false')}
          className="rounded"
        />
        <span className="text-sm text-slate-700 dark:text-slate-300">
          Dry run — report what would be deleted, delete nothing
        </span>
      </label>
      <p className="text-xs text-slate-500 -mt-4 ml-6">
        Leave this on until you have reviewed a report. Turning it off makes the
        next scheduled run delete files.
      </p>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={isTruthy(keepBest)}
          onChange={(e) => setKeepBest(e.target.checked ? 'true' : 'false')}
          className="rounded"
        />
        <span className="text-sm text-slate-700 dark:text-slate-300">
          Always keep the best (lowest-loss) checkpoint
        </span>
      </label>

      <div>
        <label className="block text-sm text-slate-700 dark:text-slate-300 mb-1">
          Keep most recent steps
        </label>
        <input
          type="number"
          min={1}
          max={50}
          value={keepLast}
          onChange={(e) => setKeepLast(e.target.value)}
          className="w-32 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white"
        />
        <p className="text-xs text-slate-500 mt-1">
          How many of the newest checkpoint steps to preserve per training. The
          newest is always kept so a run stays resumable.
        </p>
      </div>

      <div>
        <label className="block text-sm text-slate-700 dark:text-slate-300 mb-1">
          Minimum age (hours)
        </label>
        <input
          type="number"
          min={0}
          max={8760}
          value={minAgeHours}
          onChange={(e) => setMinAgeHours(e.target.value)}
          className="w-32 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white"
        />
        <p className="text-xs text-slate-500 mt-1">
          Checkpoints younger than this are never pruned.
        </p>
      </div>

      <div className="border-t border-slate-200 dark:border-slate-800 pt-4">
        <h4 className="text-sm font-medium text-slate-900 dark:text-white mb-1">
          Run cleanup now (all trainings)
        </h4>
        <p className="text-xs text-slate-500 mb-3">
          Applies the policy above to every training immediately, without waiting
          for the daily sweep, and runs even while the schedule is disabled. The
          same guards apply — the best checkpoint, the newest {keepLast || '2'},
          anything under {minAgeHours || '24'}h, and any training that could still
          resume are never touched.
        </p>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={loadSweepPreview}
            disabled={sweepBusy}
            className="px-3 py-1.5 text-sm rounded border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50"
          >
            {sweepBusy ? 'Working…' : 'Preview all trainings'}
          </button>
          <button
            type="button"
            onClick={runSweep}
            disabled={sweepBusy}
            className={`px-4 py-1.5 text-sm rounded text-white transition-colors disabled:opacity-50 ${
              isTruthy(dryRun)
                ? 'bg-slate-600 hover:bg-slate-700'
                : 'bg-red-600 hover:bg-red-700'
            }`}
          >
            {isTruthy(dryRun) ? 'Run cleanup now (dry run)' : 'Run cleanup now'}
          </button>
          {sweepMsg && <span className="text-sm text-emerald-500">{sweepMsg}</span>}
        </div>

        {sweepPreview && (
          <div
            role="status"
            className="mt-3 p-2 rounded border border-slate-300 dark:border-slate-700 text-xs text-slate-700 dark:text-slate-300"
          >
            Would remove{' '}
            <span className="font-semibold">{sweepPreview.total_checkpoints}</span>{' '}
            checkpoint{sweepPreview.total_checkpoints === 1 ? '' : 's'} across{' '}
            <span className="font-semibold">{sweepPreview.trainings_affected}</span>{' '}
            training{sweepPreview.trainings_affected === 1 ? '' : 's'}, reclaiming{' '}
            <span className="font-semibold">{gb(sweepPreview.estimated_bytes)}</span>.
            {sweepPreview.total_checkpoints === 0 && ' Nothing is eligible right now.'}
          </div>
        )}

        {sweepError && (
          <div role="alert" className="mt-3 text-xs text-red-400">
            {sweepError}
          </div>
        )}
      </div>

      <CheckpointPrunePreviewPanel />

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={async () => {
            // ORDER MATTERS: write every RESTRICTIVE setting before `enabled`.
            // Saving `enabled` first means a mid-sequence failure leaves
            // scheduled pruning switched ON with the previous (possibly
            // dry_run=false) safety settings — deleting files against a config
            // the operator never actually confirmed.
            setSaveError(null);
            try {
              await handleSave('checkpoint_prune_dry_run', dryRun);
              await handleSave('checkpoint_prune_keep_last', keepLast);
              await handleSave('checkpoint_prune_keep_best', keepBest);
              await handleSave('checkpoint_prune_min_age_hours', minAgeHours);
              await handleSave('checkpoint_prune_enabled', enabled);
              setToast('Saved');
              setTimeout(() => setToast(null), 2000);
            } catch (e: any) {
              // Never claim success on a partial write.
              setSaveError(
                e?.response?.data?.detail ||
                  'Failed to save settings — some values may not have been applied'
              );
            }
          }}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded transition-colors"
        >
          Save
        </button>
        {toast && <span className="text-sm text-emerald-500">{toast}</span>}
        {saveError && (
          <span role="alert" className="text-sm text-red-500">
            {saveError}
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Dry-run report for checkpoint pruning.
 *
 * The Storage tab tells operators to review a report before enabling deletion —
 * this is that report. Without it the only way to see what pruning would remove
 * is the Celery worker log, which makes the advice unfollowable.
 */
// Exported for MIS-E2E-128's regression test. The prune confirmation is the
// product's only guard on an irreversible deletion, so it is tested directly
// rather than through the whole settings page.
export function CheckpointPrunePreviewPanel() {
  const { trainings, fetchTrainings, previewCheckpointPrune, pruneCheckpoints } =
    useTrainingsStore();
  const [selected, setSelected] = useState<string>('');
  const [preview, setPreview] = useState<CheckpointPrunePreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pruning, setPruning] = useState(false);
  const [pruneMsg, setPruneMsg] = useState<string | null>(null);

  useEffect(() => {
    if (trainings.length === 0) {
      fetchTrainings().catch(() => {
        /* listing failures are surfaced by the trainings panel itself */
      });
    }
  }, [trainings.length, fetchTrainings]);

  const runPreview = async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setPreview(null);
    try {
      setPreview(await previewCheckpointPrune(selected));
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to load prune preview');
    } finally {
      setLoading(false);
    }
  };

  const gb = (bytes: number) => (bytes / 1e9).toFixed(2);

  return (
    <div className="border-t border-slate-300 dark:border-slate-700 pt-4 space-y-3">
      <h4 className="text-sm font-medium text-slate-900 dark:text-white">
        Preview (read-only)
      </h4>
      <p className="text-xs text-slate-500">
        Shows exactly which checkpoint steps the current policy would delete for
        a training. Nothing is removed by running a preview.
      </p>

      <div className="flex items-center gap-2">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          aria-label="Training to preview"
          className="flex-1 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white text-sm"
        >
          <option value="">Select a training…</option>
          {trainings.map((t) => (
            <option key={t.id} value={t.id}>
              {t.id} ({t.status})
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={runPreview}
          disabled={!selected || loading}
          className="px-4 py-2 bg-slate-600 hover:bg-slate-700 disabled:opacity-50 text-white rounded text-sm transition-colors"
        >
          {loading ? 'Checking…' : 'Preview'}
        </button>
      </div>

      {error && <p className="text-xs text-red-500">{error}</p>}

      {preview && !preview.skipped_reason && preview.checkpoint_count > 0 && (
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={async () => {
              // MIS-E2E-128. `preview` is a SNAPSHOT taken when Preview was
              // clicked; the Celery task re-reads `checkpoint_prune_dry_run`
              // from settings at execution time. Untick "dry run", Save, then
              // Prune now, and the dialog said "This will report on 12
              // checkpoint file(s)" while the task permanently deleted all
              // twelve. A confirmation for an irreversible action must be
              // rendered from the state the action will use — never a snapshot.
              setPruning(true);
              setPruneMsg(null);
              setError(null);

              let current: CheckpointPrunePreview;
              try {
                current = await previewCheckpointPrune(selected);
                setPreview(current);
              } catch (e: any) {
                setError(
                  e?.response?.data?.detail ||
                    'Could not confirm the current prune policy — nothing was pruned'
                );
                setPruning(false);
                return;
              }

              // Fail CLOSED. If the refreshed policy cannot be read, treat it
              // as destructive rather than assuming dry-run: for `dry_run`,
              // false is the deleting value, so the safe default is the loud
              // one. (Feature 21's recorded lesson, same setting.)
              const isDryRun = current.policy?.dry_run === true;
              const verb = isDryRun ? 'report on' : 'PERMANENTLY DELETE';

              if (
                !confirm(
                  `This will ${verb} ${current.checkpoint_count} checkpoint file(s) ` +
                    `for ${selected}. Continue?`
                )
              ) {
                setPruning(false);
                return;
              }

              try {
                await pruneCheckpoints(selected);
                setPruneMsg(
                  isDryRun
                    ? 'Dry-run prune queued — check the worker log for the report.'
                    : 'Prune queued.'
                );
              } catch (e: any) {
                setError(e?.response?.data?.detail || 'Failed to start prune');
              } finally {
                setPruning(false);
              }
            }}
            disabled={pruning}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white rounded text-sm transition-colors"
          >
            {pruning ? 'Starting…' : 'Prune now'}
          </button>
          {pruneMsg && <span className="text-xs text-emerald-500">{pruneMsg}</span>}
        </div>
      )}

      {preview && (
        <div className="text-xs text-slate-600 dark:text-slate-300 space-y-1 bg-slate-100 dark:bg-slate-800/60 rounded p-3">
          {preview.skipped_reason ? (
            <p>
              Skipped: <span className="font-medium">{preview.skipped_reason}</span>
            </p>
          ) : (
            <>
              <p>
                Would delete{' '}
                <span className="font-medium text-red-500">
                  {preview.checkpoint_count} checkpoint file(s)
                </span>{' '}
                across steps [{preview.prunable_steps.join(', ') || '—'}], freeing
                ~{gb(preview.estimated_bytes)} GB.
              </p>
              <p>Keeping steps: [{preview.kept_steps.join(', ') || '—'}]</p>
              {preview.policy.dry_run && (
                <p className="text-amber-600 dark:text-amber-400">
                  Dry run is ON — a real prune would delete nothing until you turn
                  it off.
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
