/**
 * LabelingPromptTemplatesPanel - Main labeling prompt template management panel.
 *
 * This component provides the primary interface for managing labeling prompt templates,
 * including CRUD operations and default template selection.
 */

import { useEffect, useState, useRef } from 'react';
import { Plus, Star, Trash2, Edit2, Check, Download, Upload, AlertCircle, Eye } from 'lucide-react';
import { useLabelingPromptTemplatesStore } from '../../stores/labelingPromptTemplatesStore';
import {
  LabelingPromptTemplate,
  LabelingPromptTemplateCreate,
  LabelingPromptTemplateUpdate,
  LabelingPromptTemplateConstraints,
} from '../../types/labelingPromptTemplate';
import {
  exportLabelingPromptTemplates,
  importLabelingPromptTemplates,
  getLabelingPromptTemplateUsageCount
} from '../../api/labelingPromptTemplates';

type TabType = 'all' | 'create';

export function LabelingPromptTemplatesPanel() {
  const {
    templates,
    defaultTemplate,
    loading,
    error,
    fetchTemplates,
    fetchDefaultTemplate,
    createTemplate,
    updateTemplate,
    deleteTemplate,
    setDefaultTemplate,
    clearError,
  } = useLabelingPromptTemplatesStore();

  const [activeTab, setActiveTab] = useState<TabType>('all');
  const [editingTemplate, setEditingTemplate] = useState<LabelingPromptTemplate | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [formData, setFormData] = useState<Partial<LabelingPromptTemplateCreate>>({
    name: '',
    description: '',
    system_message: '',
    user_prompt_template: '',
    temperature: 0.3,
    max_tokens: 50,
    top_p: 0.9,
    is_default: false,
  });
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [showImportModal, setShowImportModal] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importResults, setImportResults] = useState<{
    success: boolean;
    message: string;
    imported_count: number;
    skipped_count: number;
    overwritten_count: number;
    failed_count: number;
    details: string[];
  } | null>(null);
  const [overwriteDuplicates, setOverwriteDuplicates] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [usageCounts, setUsageCounts] = useState<Record<string, number>>({});
  const [previewTemplate, setPreviewTemplate] = useState<LabelingPromptTemplate | null>(null);

  // Fetch templates on mount
  useEffect(() => {
    fetchTemplates();
    fetchDefaultTemplate();
  }, [fetchTemplates, fetchDefaultTemplate]);

  // Fetch usage counts when templates change
  useEffect(() => {
    const fetchUsageCounts = async () => {
      const counts: Record<string, number> = {};

      // Fetch usage counts for all templates in parallel
      await Promise.all(
        templates.map(async (template) => {
          try {
            const result = await getLabelingPromptTemplateUsageCount(template.id);
            counts[template.id] = result.usage_count;
          } catch (err) {
            console.error(`Failed to fetch usage count for template ${template.id}:`, err);
            counts[template.id] = 0;
          }
        })
      );

      setUsageCounts(counts);
    };

    if (templates.length > 0) {
      fetchUsageCounts();
    }
  }, [templates]);

  // Show notification helper
  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  // Reset form
  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      system_message: '',
      user_prompt_template: '',
      temperature: 0.3,
      max_tokens: 50,
      top_p: 0.9,
      is_default: false,
    });
    setEditingTemplate(null);
  };

  // Handle create template
  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      // Validation
      if (!formData.user_prompt_template?.includes('{tokens_table}')) {
        showNotification('error', 'User prompt template must contain {tokens_table} placeholder');
        return;
      }

      await createTemplate(formData as LabelingPromptTemplateCreate);
      showNotification('success', 'Template created successfully');
      setActiveTab('all');
      resetForm();
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to create template');
    }
  };

  // Handle update template
  const handleUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingTemplate) return;

    try {
      // Validation
      if (formData.user_prompt_template && !formData.user_prompt_template.includes('{tokens_table}')) {
        showNotification('error', 'User prompt template must contain {tokens_table} placeholder');
        return;
      }

      await updateTemplate(editingTemplate.id, formData as LabelingPromptTemplateUpdate);
      showNotification('success', 'Template updated successfully');
      setShowEditModal(false);
      resetForm();
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to update template');
    }
  };

  // Handle delete template
  const handleDelete = async (id: string) => {
    try {
      await deleteTemplate(id);
      showNotification('success', 'Template deleted successfully');
      setShowDeleteConfirm(null);
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to delete template');
    }
  };

  // Handle set default
  const handleSetDefault = async (id: string) => {
    try {
      await setDefaultTemplate(id);
      showNotification('success', 'Default template updated');
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to set default');
    }
  };

  // Handle edit template
  const handleEdit = (template: LabelingPromptTemplate) => {
    setEditingTemplate(template);
    setFormData({
      name: template.name,
      description: template.description,
      system_message: template.system_message,
      user_prompt_template: template.user_prompt_template,
      temperature: template.temperature,
      max_tokens: template.max_tokens,
      top_p: template.top_p,
    });
    setShowEditModal(true);
  };

  // Handle export templates
  const handleExport = async () => {
    try {
      setIsExporting(true);

      // Get only custom template IDs (exclude system templates)
      const customTemplateIds = templates
        .filter(t => !t.is_system)
        .map(t => t.id);

      if (customTemplateIds.length === 0) {
        showNotification('error', 'No custom templates to export');
        return;
      }

      const exportData = await exportLabelingPromptTemplates(customTemplateIds);

      // Create download
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `labeling-templates-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      showNotification('success', `Exported ${exportData.templates.length} template(s)`);
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to export templates');
    } finally {
      setIsExporting(false);
    }
  };

  // Handle import file selection
  const handleImportFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImportFile(file);
      setShowImportModal(true);
    }
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle import templates
  const handleImport = async () => {
    if (!importFile) return;

    try {
      setIsImporting(true);

      // Read file
      const fileContent = await importFile.text();
      const importData = JSON.parse(fileContent);

      // Validate format
      if (!importData.version || !importData.templates || !Array.isArray(importData.templates)) {
        showNotification('error', 'Invalid import file format');
        return;
      }

      // Import templates
      const result = await importLabelingPromptTemplates(importData, overwriteDuplicates);

      setImportResults(result);

      // Refresh templates list
      await fetchTemplates();

      if (result.success) {
        showNotification('success', result.message);
      } else {
        showNotification('error', result.message);
      }
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to import templates');
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-slate-900">
      {/* Header */}
      <div className="border-b border-slate-700 p-4">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold text-slate-100">Labeling Prompt Templates</h2>
            <p className="text-sm text-slate-400 mt-1">
              Manage prompt templates for semantic feature labeling
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              disabled={isExporting || templates.filter(t => !t.is_system).length === 0}
              className="inline-flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded transition-colors text-sm"
              title="Export custom templates"
            >
              <Download className="w-4 h-4" />
              {isExporting ? 'Exporting...' : 'Export'}
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isImporting}
              className="inline-flex items-center gap-2 px-3 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded transition-colors text-sm"
              title="Import templates"
            >
              <Upload className="w-4 h-4" />
              {isImporting ? 'Importing...' : 'Import'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImportFileSelect}
              className="hidden"
            />
          </div>
        </div>
      </div>

      {/* Notification */}
      {notification && (
        <div
          className={`mx-4 mt-4 p-3 rounded ${
            notification.type === 'success'
              ? 'bg-emerald-900/50 text-emerald-200 border border-emerald-700'
              : 'bg-red-900/50 text-red-200 border border-red-700'
          }`}
        >
          {notification.message}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mx-4 mt-4 p-3 rounded bg-red-900/50 text-red-200 border border-red-700">
          {error}
          <button onClick={clearError} className="ml-2 underline">
            Dismiss
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-4">
        <div className="flex space-x-4">
          <button
            onClick={() => {
              setActiveTab('all');
              resetForm();
              setShowEditModal(false);
            }}
            className={`py-2 px-4 font-medium transition-colors ${
              activeTab === 'all'
                ? 'text-emerald-400 border-b-2 border-emerald-400'
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            All Templates ({templates.length})
          </button>
          <button
            onClick={() => {
              setActiveTab('create');
              resetForm();
              setShowEditModal(false);
            }}
            className={`py-2 px-4 font-medium transition-colors ${
              activeTab === 'create'
                ? 'text-emerald-400 border-b-2 border-emerald-400'
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            <Plus className="w-4 h-4 inline mr-1" />
            Create New
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {activeTab === 'all' && (
          <div className="space-y-3">
            {loading ? (
              <div className="text-center py-8 text-slate-400">Loading templates...</div>
            ) : templates.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                No templates found. Create your first template to get started.
              </div>
            ) : (
              templates.map((template) => (
                <div
                  key={template.id}
                  className="bg-slate-800 rounded-lg border border-slate-700 p-4 hover:border-slate-600 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-medium text-slate-100">{template.name}</h3>
                        {template.is_default && (
                          <span className="px-2 py-1 text-xs bg-emerald-900/30 text-emerald-300 rounded border border-emerald-700 flex items-center gap-1">
                            <Check className="w-3 h-3" />
                            Default
                          </span>
                        )}
                        {template.is_system && (
                          <span className="px-2 py-1 text-xs bg-blue-900/30 text-blue-300 rounded border border-blue-700">
                            System
                          </span>
                        )}
                        {usageCounts[template.id] !== undefined && usageCounts[template.id] > 0 && (
                          <span className="px-2 py-1 text-xs bg-purple-900/30 text-purple-300 rounded border border-purple-700" title={`Used by ${usageCounts[template.id]} labeling job(s)`}>
                            {usageCounts[template.id]} {usageCounts[template.id] === 1 ? 'job' : 'jobs'}
                          </span>
                        )}
                      </div>
                      {template.description && (
                        <p className="text-sm text-slate-400 mt-1">{template.description}</p>
                      )}
                      <div className="flex gap-4 mt-2 text-xs text-slate-500">
                        <span>Temp: {template.temperature}</span>
                        <span>Max Tokens: {template.max_tokens}</span>
                        <span>Top-p: {template.top_p}</span>
                      </div>
                    </div>
                    <div className="flex gap-2 ml-4">
                      <button
                        onClick={() => setPreviewTemplate(template)}
                        className="p-2 text-slate-400 hover:text-purple-400 hover:bg-slate-700 rounded transition-colors"
                        title="Preview"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      {!template.is_default && (
                        <button
                          onClick={() => handleSetDefault(template.id)}
                          className="p-2 text-slate-400 hover:text-emerald-400 hover:bg-slate-700 rounded transition-colors"
                          title="Set as default"
                        >
                          <Star className="w-4 h-4" />
                        </button>
                      )}
                      {!template.is_system && (
                        <>
                          <button
                            onClick={() => handleEdit(template)}
                            className="p-2 text-slate-400 hover:text-blue-400 hover:bg-slate-700 rounded transition-colors"
                            title="Edit"
                          >
                            <Edit2 className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setShowDeleteConfirm(template.id)}
                            className="p-2 text-slate-400 hover:text-red-400 hover:bg-slate-700 rounded transition-colors"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Delete Confirmation */}
                  {showDeleteConfirm === template.id && (
                    <div className="mt-3 p-3 bg-red-900/20 border border-red-700 rounded">
                      <p className="text-sm text-red-300 mb-2">
                        Are you sure you want to delete this template?
                      </p>
                      {usageCounts[template.id] > 0 && (
                        <p className="text-xs text-yellow-300 mb-2 flex items-start gap-1">
                          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                          <span>
                            Warning: This template is currently used by {usageCounts[template.id]} labeling job{usageCounts[template.id] !== 1 ? 's' : ''}.
                            Deleting it may affect those jobs.
                          </span>
                        </p>
                      )}
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleDelete(template.id)}
                          className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm transition-colors"
                        >
                          Delete
                        </button>
                        <button
                          onClick={() => setShowDeleteConfirm(null)}
                          className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded text-sm transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'create' && (
          <form onSubmit={handleCreate} className="max-w-3xl space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Template Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500"
                required
                maxLength={LabelingPromptTemplateConstraints.name.maxLength}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Description
              </label>
              <input
                type="text"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                System Message *
              </label>
              <textarea
                value={formData.system_message}
                onChange={(e) => setFormData({ ...formData, system_message: e.target.value })}
                rows={3}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500 font-mono text-sm"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                User Prompt Template * (must include <code className="text-emerald-400">{'{tokens_table}'}</code>)
              </label>
              <textarea
                value={formData.user_prompt_template}
                onChange={(e) => setFormData({ ...formData, user_prompt_template: e.target.value })}
                rows={8}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500 font-mono text-sm"
                required
              />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Temperature (0-2)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min={LabelingPromptTemplateConstraints.temperature.min}
                  max={LabelingPromptTemplateConstraints.temperature.max}
                  value={formData.temperature}
                  onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Max Tokens (10-500)
                </label>
                <input
                  type="number"
                  min={LabelingPromptTemplateConstraints.max_tokens.min}
                  max={LabelingPromptTemplateConstraints.max_tokens.max}
                  value={formData.max_tokens}
                  onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Top-p (0-1)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min={LabelingPromptTemplateConstraints.top_p.min}
                  max={LabelingPromptTemplateConstraints.top_p.max}
                  value={formData.top_p}
                  onChange={(e) => setFormData({ ...formData, top_p: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="is_default"
                checked={formData.is_default}
                onChange={(e) => setFormData({ ...formData, is_default: e.target.checked })}
                className="rounded bg-slate-800 border-slate-700"
              />
              <label htmlFor="is_default" className="text-sm text-slate-300">
                Set as default template
              </label>
            </div>

            <div className="flex gap-3 pt-4">
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Creating...' : 'Create Template'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setActiveTab('all');
                  resetForm();
                }}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        )}
      </div>

      {/* Edit Modal */}
      {showEditModal && editingTemplate && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 rounded-lg max-w-3xl w-full max-h-[90vh] overflow-auto">
            <div className="p-4 border-b border-slate-700 flex justify-between items-center sticky top-0 bg-slate-800">
              <h3 className="text-lg font-semibold text-slate-100">Edit Template</h3>
              <button
                onClick={() => {
                  setShowEditModal(false);
                  resetForm();
                }}
                className="text-slate-400 hover:text-slate-200"
              >
                ×
              </button>
            </div>
            <form onSubmit={handleUpdate} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Description</label>
                <input
                  type="text"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">System Message</label>
                <textarea
                  value={formData.system_message}
                  onChange={(e) => setFormData({ ...formData, system_message: e.target.value })}
                  rows={3}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100 font-mono text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">User Prompt Template</label>
                <textarea
                  value={formData.user_prompt_template}
                  onChange={(e) => setFormData({ ...formData, user_prompt_template: e.target.value })}
                  rows={8}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100 font-mono text-sm"
                />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Temperature</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={formData.temperature}
                    onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Max Tokens</label>
                  <input
                    type="number"
                    min="10"
                    max="500"
                    value={formData.max_tokens}
                    onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Top-p</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={formData.top_p}
                    onChange={(e) => setFormData({ ...formData, top_p: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100"
                  />
                </div>
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded font-medium transition-colors"
                >
                  {loading ? 'Saving...' : 'Save Changes'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowEditModal(false);
                    resetForm();
                  }}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded font-medium transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Preview Modal */}
      {previewTemplate && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="p-4 border-b border-slate-700 flex justify-between items-center sticky top-0 bg-slate-800">
              <div>
                <h3 className="text-lg font-semibold text-slate-100">Template Preview</h3>
                <p className="text-sm text-slate-400 mt-1">{previewTemplate.name}</p>
              </div>
              <button
                onClick={() => setPreviewTemplate(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                ×
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Description */}
              {previewTemplate.description && (
                <div>
                  <p className="text-sm text-slate-300">{previewTemplate.description}</p>
                </div>
              )}

              {/* Parameters */}
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="bg-slate-900/50 p-3 rounded">
                  <span className="text-slate-400">Temperature:</span>
                  <span className="ml-2 text-slate-200 font-medium">{previewTemplate.temperature}</span>
                </div>
                <div className="bg-slate-900/50 p-3 rounded">
                  <span className="text-slate-400">Max Tokens:</span>
                  <span className="ml-2 text-slate-200 font-medium">{previewTemplate.max_tokens}</span>
                </div>
                <div className="bg-slate-900/50 p-3 rounded">
                  <span className="text-slate-400">Top-p:</span>
                  <span className="ml-2 text-slate-200 font-medium">{previewTemplate.top_p}</span>
                </div>
              </div>

              {/* System Message */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-2">System Message</h4>
                <div className="bg-slate-900/50 p-3 rounded border border-slate-700">
                  <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap">{previewTemplate.system_message}</pre>
                </div>
              </div>

              {/* User Prompt with Sample Data */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-2">
                  User Prompt Template
                  <span className="ml-2 text-xs text-slate-500">(with sample token data)</span>
                </h4>
                <div className="bg-slate-900/50 p-3 rounded border border-slate-700">
                  <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap">
                    {previewTemplate.user_prompt_template.replace(
                      '{tokens_table}',
                      `Token #1: "the" → Activation: 0.89\nToken #2: "quick" → Activation: 0.76\nToken #3: "brown" → Activation: 0.82\nToken #4: "fox" → Activation: 0.91\nToken #5: "jumps" → Activation: 0.68`
                    )}
                  </pre>
                </div>
              </div>

              {/* Example Output */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-2">
                  Example Expected Output
                  <span className="ml-2 text-xs text-slate-500">(illustrative)</span>
                </h4>
                <div className="bg-emerald-900/10 border border-emerald-700/30 p-3 rounded">
                  <p className="text-xs text-emerald-200/80 font-mono">
                    Animal-related tokens and motion verbs
                  </p>
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  Note: Actual output will vary based on the model and input tokens.
                </p>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setPreviewTemplate(null)}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded font-medium transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && importFile && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 rounded-lg max-w-2xl w-full">
            <div className="p-4 border-b border-slate-700 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-slate-100">Import Templates</h3>
              <button
                onClick={() => {
                  setShowImportModal(false);
                  setImportFile(null);
                  setImportResults(null);
                  setOverwriteDuplicates(false);
                }}
                className="text-slate-400 hover:text-slate-200"
              >
                ×
              </button>
            </div>

            <div className="p-4 space-y-4">
              <div>
                <p className="text-sm text-slate-300 mb-2">
                  File: <span className="font-mono text-emerald-400">{importFile.name}</span>
                </p>
                <p className="text-xs text-slate-400">
                  Size: {(importFile.size / 1024).toFixed(2)} KB
                </p>
              </div>

              {!importResults ? (
                <>
                  <div className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      id="overwrite"
                      checked={overwriteDuplicates}
                      onChange={(e) => setOverwriteDuplicates(e.target.checked)}
                      className="mt-1 rounded bg-slate-700 border-slate-600"
                    />
                    <label htmlFor="overwrite" className="text-sm text-slate-300">
                      <span className="font-medium">Overwrite duplicate templates</span>
                      <p className="text-xs text-slate-400 mt-1">
                        If enabled, templates with the same name will be updated. Otherwise, duplicates will be skipped.
                      </p>
                    </label>
                  </div>

                  <div className="flex gap-3 pt-4">
                    <button
                      onClick={handleImport}
                      disabled={isImporting}
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded font-medium transition-colors"
                    >
                      {isImporting ? 'Importing...' : 'Import Templates'}
                    </button>
                    <button
                      onClick={() => {
                        setShowImportModal(false);
                        setImportFile(null);
                        setOverwriteDuplicates(false);
                      }}
                      disabled={isImporting}
                      className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded font-medium transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className={`p-3 rounded border ${
                    importResults.success
                      ? 'bg-emerald-900/20 border-emerald-700'
                      : 'bg-red-900/20 border-red-700'
                  }`}>
                    <p className={`text-sm font-medium ${
                      importResults.success ? 'text-emerald-200' : 'text-red-200'
                    }`}>
                      {importResults.message}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-slate-900/50 p-3 rounded">
                      <span className="text-slate-400">Imported:</span>
                      <span className="ml-2 text-emerald-400 font-medium">{importResults.imported_count}</span>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded">
                      <span className="text-slate-400">Overwritten:</span>
                      <span className="ml-2 text-blue-400 font-medium">{importResults.overwritten_count}</span>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded">
                      <span className="text-slate-400">Skipped:</span>
                      <span className="ml-2 text-yellow-400 font-medium">{importResults.skipped_count}</span>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded">
                      <span className="text-slate-400">Failed:</span>
                      <span className="ml-2 text-red-400 font-medium">{importResults.failed_count}</span>
                    </div>
                  </div>

                  {importResults.details.length > 0 && (
                    <div className="max-h-48 overflow-auto">
                      <p className="text-xs font-medium text-slate-400 mb-2">Details:</p>
                      <div className="space-y-1">
                        {importResults.details.map((detail, index) => (
                          <p key={index} className="text-xs text-slate-300 font-mono bg-slate-900/50 p-2 rounded">
                            {detail}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-3 pt-4">
                    <button
                      onClick={() => {
                        setShowImportModal(false);
                        setImportFile(null);
                        setImportResults(null);
                        setOverwriteDuplicates(false);
                      }}
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded font-medium transition-colors"
                    >
                      Done
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
