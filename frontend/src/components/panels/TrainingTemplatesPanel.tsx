/**
 * TrainingTemplatesPanel - Main training template management panel.
 *
 * This component provides the primary interface for managing training templates,
 * including CRUD operations, favorites, and export/import functionality.
 */

import { useEffect, useState, useRef } from 'react';
import { Download, Upload, Plus, X } from 'lucide-react';
import { useTrainingTemplatesStore } from '../../stores/trainingTemplatesStore';
import { TrainingTemplateList } from '../trainingTemplates/TrainingTemplateList';
import { TrainingTemplateForm } from '../trainingTemplates/TrainingTemplateForm';
import {
  TrainingTemplate,
  TrainingTemplateCreate,
  TrainingTemplateUpdate,
} from '../../types/trainingTemplate';

type TabType = 'all' | 'favorites' | 'create';

export function TrainingTemplatesPanel() {
  const {
    templates,
    favorites,
    loading,
    error,
    pagination,
    fetchTemplates,
    fetchFavorites,
    createTemplate,
    updateTemplate,
    deleteTemplate,
    toggleFavorite,
    exportTemplates,
    importTemplates,
    clearError,
  } = useTrainingTemplatesStore();

  const [activeTab, setActiveTab] = useState<TabType>('all');
  const [editingTemplate, setEditingTemplate] = useState<TrainingTemplate | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch templates on mount
  useEffect(() => {
    fetchTemplates();
    fetchFavorites();
  }, [fetchTemplates, fetchFavorites]);

  // Show notification helper
  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  // Handle create template
  const handleCreate = async (data: TrainingTemplateCreate | TrainingTemplateUpdate) => {
    try {
      await createTemplate(data as TrainingTemplateCreate);
      showNotification('success', 'Template created successfully');
      setShowEditModal(false);
      setEditingTemplate(null);
      setActiveTab('all');
      fetchFavorites(); // Refresh favorites if new template is favorite
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to create template');
    }
  };

  // Handle update template
  const handleUpdate = async (data: TrainingTemplateUpdate) => {
    if (!editingTemplate) return;
    try {
      await updateTemplate(editingTemplate.id, data);
      showNotification('success', 'Template updated successfully');
      setShowEditModal(false);
      setEditingTemplate(null);
      fetchFavorites(); // Refresh favorites if favorite status changed
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to update template');
    }
  };

  // Handle delete template
  const handleDelete = async (id: string) => {
    try {
      await deleteTemplate(id);
      showNotification('success', 'Template deleted successfully');
      fetchFavorites(); // Refresh favorites list
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to delete template');
    }
  };

  // Handle toggle favorite
  const handleToggleFavorite = async (id: string) => {
    try {
      await toggleFavorite(id);
      fetchFavorites(); // Refresh favorites list
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to toggle favorite');
    }
  };

  // Handle duplicate template
  const handleDuplicate = (template: TrainingTemplate) => {
    const duplicateData: TrainingTemplateCreate = {
      name: `${template.name} (Copy)`,
      description: template.description,
      model_id: template.model_id,
      dataset_id: template.dataset_id,
      encoder_type: template.encoder_type as any,
      hyperparameters: { ...template.hyperparameters },
      is_favorite: false,
      extra_metadata: template.extra_metadata ? { ...template.extra_metadata } : undefined,
    };
    setEditingTemplate(duplicateData as any);
    setShowEditModal(true);
  };

  // Handle edit template
  const handleEdit = (template: TrainingTemplate) => {
    setEditingTemplate(template);
    setShowEditModal(true);
  };

  // Handle export
  const handleExport = async () => {
    try {
      const exportData = await exportTemplates();
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `training-templates-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showNotification('success', `Exported ${exportData.templates.length} template(s)`);
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to export templates');
    }
  };

  // Handle import
  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleImportFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const importData = JSON.parse(text);

      // Validate import data structure
      if (!importData.templates || !Array.isArray(importData.templates)) {
        throw new Error('Invalid import file format');
      }

      const result = await importTemplates(importData, false);

      showNotification(
        'success',
        `Import successful: ${result.created} created, ${result.updated} updated, ${result.skipped} skipped`
      );

      fetchFavorites(); // Refresh favorites if imported templates are favorites
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to import templates');
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle page change
  const handlePageChange = (page: number) => {
    fetchTemplates({ page });
  };

  // Get current templates based on active tab
  const currentTemplates = activeTab === 'favorites' ? favorites : templates;

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="max-w-[80%] mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-slate-100 mb-2">Training Templates</h1>
          <p className="text-slate-400">
            Manage SAE training templates with preset hyperparameters
          </p>
        </div>

        {/* Notification */}
        {notification && (
          <div
            className={`mb-6 p-4 rounded-lg border ${
              notification.type === 'success'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-red-500/10 border-red-500/30 text-red-400'
            }`}
          >
            <div className="flex items-center justify-between">
              <span>{notification.message}</span>
              <button
                onClick={() => setNotification(null)}
                className="p-1 hover:bg-white/10 rounded transition-colors"
                title="Dismiss notification"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-red-400">{error}</span>
              <button
                onClick={clearError}
                className="p-1 hover:bg-white/10 rounded transition-colors"
                title="Dismiss error"
              >
                <X className="w-4 h-4 text-red-400" />
              </button>
            </div>
          </div>
        )}

        {/* Tabs and Actions */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('all')}
              className={`px-4 py-2 rounded transition-colors ${
                activeTab === 'all'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              All Templates ({templates.length})
            </button>
            <button
              onClick={() => setActiveTab('favorites')}
              className={`px-4 py-2 rounded transition-colors ${
                activeTab === 'favorites'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Favorites ({favorites.length})
            </button>
            <button
              onClick={() => setActiveTab('create')}
              className={`px-4 py-2 rounded transition-colors ${
                activeTab === 'create'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
              title="Create new training template"
            >
              <Plus className="w-4 h-4 inline mr-1" />
              Create New
            </button>
          </div>

          {activeTab !== 'create' && (
            <div className="flex gap-2">
              <button
                onClick={handleExport}
                disabled={templates.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded transition-colors"
                title="Export all templates"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button
                onClick={handleImportClick}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors"
                title="Import templates from JSON"
              >
                <Upload className="w-4 h-4" />
                Import
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleImportFile}
                className="hidden"
              />
            </div>
          )}
        </div>

        {/* Content based on active tab */}
        {activeTab === 'create' && (
          <TrainingTemplateForm onSubmit={handleCreate} />
        )}

        {activeTab !== 'create' && (
          <TrainingTemplateList
            templates={currentTemplates}
            loading={loading}
            onEdit={handleEdit}
            onDelete={handleDelete}
            onToggleFavorite={handleToggleFavorite}
            onDuplicate={handleDuplicate}
            currentPage={pagination?.page}
            totalPages={pagination?.total_pages}
            hasNext={pagination?.has_next}
            hasPrev={pagination?.has_prev}
            onPageChange={handlePageChange}
          />
        )}

        {/* Edit Modal */}
        {showEditModal && editingTemplate && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-slate-900 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <TrainingTemplateForm
                  template={editingTemplate}
                  onSubmit={editingTemplate.id ? handleUpdate : handleCreate}
                  onCancel={() => {
                    setShowEditModal(false);
                    setEditingTemplate(null);
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
