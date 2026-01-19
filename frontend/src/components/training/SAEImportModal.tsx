/**
 * SAE Import Modal Component
 *
 * Modal dialog for selecting which SAEs to import from a completed training.
 * Supports multi-layer/multi-hook trainings where multiple SAEs are available.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { X, Download, CheckCircle, Loader2, AlertCircle, Layers, CheckCircle2 } from 'lucide-react';
import { Training } from '../../types/training';
import {
  AvailableSAEInfo,
  ImportedSAEInfo,
  SAEImportFromTrainingResponse,
} from '../../types/sae';
import {
  getAvailableSAEsFromTraining,
  importSAEFromTraining,
} from '../../api/saes';

interface SAEImportModalProps {
  training: Training;
  isOpen: boolean;
  onClose: () => void;
  onImportComplete: (response: SAEImportFromTrainingResponse) => void;
  modelName?: string;
  datasetName?: string;
}

function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return 'Unknown';
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export const SAEImportModal: React.FC<SAEImportModalProps> = ({
  training,
  isOpen,
  onClose,
  onImportComplete,
  modelName,
  datasetName,
}) => {
  const [availableSAEs, setAvailableSAEs] = useState<AvailableSAEInfo[]>([]);
  const [importedSAEs, setImportedSAEs] = useState<ImportedSAEInfo[]>([]);
  const [selectedSAEs, setSelectedSAEs] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch available SAEs when modal opens
  useEffect(() => {
    if (isOpen && training.id) {
      setIsLoading(true);
      setError(null);

      getAvailableSAEsFromTraining(training.id)
        .then((response) => {
          setAvailableSAEs(response.available_saes);
          setImportedSAEs(response.imported_saes || []);
          // Select all available by default
          const allKeys = response.available_saes.map(
            (sae) => `${sae.layer}_${sae.hook_type}`
          );
          setSelectedSAEs(new Set(allKeys));
        })
        .catch((err) => {
          console.error('Failed to fetch available SAEs:', err);
          setError(err.message || 'Failed to load available SAEs');
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [isOpen, training.id]);

  // Calculate total size of selected SAEs
  const totalSelectedSize = useMemo(() => {
    return availableSAEs
      .filter((sae) => selectedSAEs.has(`${sae.layer}_${sae.hook_type}`))
      .reduce((sum, sae) => sum + (sae.size_bytes || 0), 0);
  }, [availableSAEs, selectedSAEs]);

  // Get unique layers and hook types
  const layers = useMemo(
    () => [...new Set(availableSAEs.map((sae) => sae.layer))].sort((a, b) => a - b),
    [availableSAEs]
  );

  const hookTypes = useMemo(
    () => [...new Set(availableSAEs.map((sae) => sae.hook_type))].sort(),
    [availableSAEs]
  );

  const toggleSAE = (layer: number, hookType: string) => {
    const key = `${layer}_${hookType}`;
    setSelectedSAEs((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const selectAll = () => {
    const allKeys = availableSAEs.map((sae) => `${sae.layer}_${sae.hook_type}`);
    setSelectedSAEs(new Set(allKeys));
  };

  const deselectAll = () => {
    setSelectedSAEs(new Set());
  };

  const handleImport = async () => {
    if (selectedSAEs.size === 0) return;

    setIsImporting(true);
    setError(null);

    try {
      // Determine if we're importing all or specific SAEs
      const importAll = selectedSAEs.size === availableSAEs.length;

      let request;
      if (importAll) {
        request = {
          training_id: training.id,
          name: modelName ? `SAE from ${modelName}` : undefined,
          description: datasetName
            ? `SAE trained on ${datasetName} using ${training.hyperparameters?.architecture_type || 'standard'} architecture`
            : undefined,
          import_all: true,
        };
      } else {
        // Extract selected layers and hook_types
        const selectedLayers = new Set<number>();
        const selectedHookTypes = new Set<string>();

        availableSAEs
          .filter((sae) => selectedSAEs.has(`${sae.layer}_${sae.hook_type}`))
          .forEach((sae) => {
            selectedLayers.add(sae.layer);
            selectedHookTypes.add(sae.hook_type);
          });

        request = {
          training_id: training.id,
          name: modelName ? `SAE from ${modelName}` : undefined,
          description: datasetName
            ? `SAE trained on ${datasetName} using ${training.hyperparameters?.architecture_type || 'standard'} architecture`
            : undefined,
          import_all: false,
          layers: [...selectedLayers],
          hook_types: [...selectedHookTypes],
        };
      }

      const response = await importSAEFromTraining(request);
      onImportComplete(response);
    } catch (err: any) {
      console.error('Failed to import SAEs:', err);
      setError(err.message || 'Failed to import SAEs');
    } finally {
      setIsImporting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-slate-900 rounded-lg shadow-xl border border-slate-700 w-full max-w-lg mx-4 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-500" />
            <h2 className="text-lg font-semibold text-white">Import SAEs</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-700 rounded transition-colors"
            disabled={isImporting}
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-emerald-500" />
              <span className="ml-2 text-slate-400">Loading available SAEs...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-8 text-red-400">
              <AlertCircle className="w-5 h-5 mr-2" />
              {error}
            </div>
          ) : availableSAEs.length === 0 && importedSAEs.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              No SAEs found in this training checkpoint.
            </div>
          ) : (
            <div className="space-y-4">
              {/* Summary */}
              <div className="text-sm text-slate-400">
                {availableSAEs.length > 0 ? (
                  <>
                    Found {availableSAEs.length} SAE(s) available for import
                    {layers.length > 0 && ` across ${layers.length} layer(s)`}
                    {hookTypes.length > 1 && ` and ${hookTypes.length} hook types`}
                  </>
                ) : (
                  'All SAEs from this training have been imported'
                )}
                {importedSAEs.length > 0 && (
                  <span className="text-slate-500">
                    {' '}({importedSAEs.length} already imported)
                  </span>
                )}
              </div>

              {/* Selection buttons - only show if there are available SAEs */}
              {availableSAEs.length > 0 && (
                <div className="flex gap-2">
                  <button
                    onClick={selectAll}
                    className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
                  >
                    Select All
                  </button>
                  <button
                    onClick={deselectAll}
                    className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
                  >
                    Deselect All
                  </button>
                </div>
              )}

              {/* SAE selection grid */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-300">
                  {availableSAEs.length > 0
                    ? `Select SAEs to Import (${selectedSAEs.size} selected)`
                    : 'SAEs in this Training'}
                </label>
                <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto pr-1">
                  {/* Available SAEs (selectable) */}
                  {availableSAEs.map((sae) => {
                    const key = `${sae.layer}_${sae.hook_type}`;
                    const isSelected = selectedSAEs.has(key);
                    return (
                      <button
                        key={key}
                        onClick={() => toggleSAE(sae.layer, sae.hook_type)}
                        className={`p-3 text-left rounded border transition-colors ${
                          isSelected
                            ? 'bg-emerald-600/20 border-emerald-500 text-white'
                            : 'bg-slate-800 border-slate-600 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">Layer {sae.layer}</span>
                          {isSelected && (
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                          )}
                        </div>
                        <div className="text-xs text-slate-400 mt-1 truncate">
                          {sae.hook_type.replace('hook_', '')}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">
                          {formatBytes(sae.size_bytes)}
                        </div>
                      </button>
                    );
                  })}
                  {/* Imported SAEs (greyed out, not selectable) */}
                  {importedSAEs.map((sae) => {
                    const key = `imported_${sae.layer}_${sae.hook_type}`;
                    return (
                      <div
                        key={key}
                        className="p-3 text-left rounded border bg-slate-800/40 border-slate-700 opacity-60 cursor-not-allowed"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-slate-500">Layer {sae.layer}</span>
                          <CheckCircle2 className="w-4 h-4 text-slate-500" />
                        </div>
                        <div className="text-xs text-slate-500 mt-1 truncate">
                          {sae.hook_type.replace('hook_', '')}
                        </div>
                        <div className="text-xs text-slate-600 mt-0.5">
                          Already imported
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Total size */}
              {selectedSAEs.size > 0 && (
                <div className="text-sm text-slate-400">
                  Total size: {formatBytes(totalSelectedSize)}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-4 border-t border-slate-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
            disabled={isImporting}
          >
            Cancel
          </button>
          <button
            onClick={handleImport}
            disabled={isImporting || selectedSAEs.size === 0}
            className={`px-4 py-2 text-sm rounded transition-colors flex items-center gap-2 ${
              isImporting || selectedSAEs.size === 0
                ? 'bg-emerald-600/50 text-white/50 cursor-not-allowed'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white'
            }`}
          >
            {isImporting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                Import {selectedSAEs.size} SAE{selectedSAEs.size !== 1 ? 's' : ''}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SAEImportModal;
