/**
 * PromptTemplatesPanel - Main prompt template management panel.
 *
 * This component provides the primary interface for managing prompt templates,
 * including CRUD operations, favorites, and export/import functionality.
 */

import { useEffect, useState, useRef } from 'react';
import { Download, Upload, Plus, X, Search, Star, Trash2, Copy, Edit2, FileText } from 'lucide-react';
import { usePromptTemplatesStore } from '../../stores/promptTemplatesStore';
import { PromptTemplate } from '../../types/promptTemplate';
import { COMPONENTS } from '../../config/brand';

type TabType = 'all' | 'favorites' | 'create';

export function PromptTemplatesPanel() {
  const {
    templates,
    favorites,
    loading,
    fetchTemplates,
    fetchFavorites,
    createTemplate,
    updateTemplate,
    deleteTemplate,
    toggleFavorite,
    duplicateTemplate,
    exportTemplates,
    importTemplates,
  } = usePromptTemplatesStore();

  const [activeTab, setActiveTab] = useState<TabType>('all');
  const [editingTemplate, setEditingTemplate] = useState<PromptTemplate | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Form state for create/edit
  const [formData, setFormData] = useState<{
    name: string;
    description: string;
    prompts: string[];
    tags: string[];
    is_favorite: boolean;
  }>({
    name: '',
    description: '',
    prompts: [''],
    tags: [],
    is_favorite: false,
  });

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

  // Reset form
  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      prompts: [''],
      tags: [],
      is_favorite: false,
    });
  };

  // Handle create template
  const handleCreate = async () => {
    if (!formData.name.trim()) {
      showNotification('error', 'Template name is required');
      return;
    }
    const validPrompts = formData.prompts.filter(p => p.trim());
    if (validPrompts.length === 0) {
      showNotification('error', 'At least one prompt is required');
      return;
    }
    try {
      await createTemplate({
        name: formData.name.trim(),
        description: formData.description.trim() || undefined,
        prompts: validPrompts,
        tags: formData.tags,
        is_favorite: formData.is_favorite,
      });
      showNotification('success', 'Template created successfully');
      resetForm();
      setActiveTab('all');
      fetchFavorites();
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to create template');
    }
  };

  // Handle update template
  const handleUpdate = async () => {
    if (!editingTemplate) return;
    if (!formData.name.trim()) {
      showNotification('error', 'Template name is required');
      return;
    }
    const validPrompts = formData.prompts.filter(p => p.trim());
    if (validPrompts.length === 0) {
      showNotification('error', 'At least one prompt is required');
      return;
    }
    try {
      await updateTemplate(editingTemplate.id, {
        name: formData.name.trim(),
        description: formData.description.trim() || undefined,
        prompts: validPrompts,
        tags: formData.tags,
        is_favorite: formData.is_favorite,
      });
      showNotification('success', 'Template updated successfully');
      setShowEditModal(false);
      setEditingTemplate(null);
      fetchFavorites();
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to update template');
    }
  };

  // Handle delete template
  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this template?')) return;
    try {
      await deleteTemplate(id);
      showNotification('success', 'Template deleted successfully');
      fetchFavorites();
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to delete template');
    }
  };

  // Handle toggle favorite
  const handleToggleFavorite = async (id: string) => {
    try {
      await toggleFavorite(id);
      fetchFavorites();
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to toggle favorite');
    }
  };

  // Handle duplicate template
  const handleDuplicate = async (id: string) => {
    try {
      await duplicateTemplate(id);
      showNotification('success', 'Template duplicated successfully');
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to duplicate template');
    }
  };

  // Handle edit template
  const handleEdit = (template: PromptTemplate) => {
    setEditingTemplate(template);
    setFormData({
      name: template.name,
      description: template.description || '',
      prompts: template.prompts.length > 0 ? [...template.prompts] : [''],
      tags: template.tags || [],
      is_favorite: template.is_favorite,
    });
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
      a.download = `prompt-templates-${new Date().toISOString().split('T')[0]}.json`;
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
  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const importData = JSON.parse(text);
      const result = await importTemplates(importData, false);
      showNotification(
        'success',
        `Imported ${result.created} template(s), ${result.skipped} skipped`
      );
    } catch (err) {
      showNotification('error', err instanceof Error ? err.message : 'Failed to import templates');
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Add prompt to form
  const addPrompt = () => {
    setFormData(prev => ({ ...prev, prompts: [...prev.prompts, ''] }));
  };

  // Remove prompt from form
  const removePrompt = (index: number) => {
    if (formData.prompts.length <= 1) return;
    setFormData(prev => ({
      ...prev,
      prompts: prev.prompts.filter((_, i) => i !== index),
    }));
  };

  // Update prompt in form
  const updatePrompt = (index: number, value: string) => {
    setFormData(prev => ({
      ...prev,
      prompts: prev.prompts.map((p, i) => (i === index ? value : p)),
    }));
  };

  // Filter templates by search
  const filteredTemplates = (activeTab === 'favorites' ? favorites : templates).filter(
    t =>
      t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 p-6 border-b border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-xl font-semibold text-slate-100">Steering Prompt Templates</h1>
            <p className="text-sm text-slate-400 mt-1">
              Save and manage prompt series for steering experiments
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleExport}
              className={`px-3 py-2 flex items-center gap-2 text-sm ${COMPONENTS.button.ghost}`}
              title="Export all templates"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className={`px-3 py-2 flex items-center gap-2 text-sm ${COMPONENTS.button.ghost}`}
              title="Import templates from file"
            >
              <Upload className="w-4 h-4" />
              Import
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />
            <button
              onClick={() => {
                resetForm();
                setActiveTab('create');
              }}
              className={`px-3 py-2 flex items-center gap-2 text-sm ${COMPONENTS.button.primary}`}
            >
              <Plus className="w-4 h-4" />
              New Template
            </button>
          </div>
        </div>

        {/* Tabs and Search */}
        <div className="flex items-center justify-between">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('all')}
              className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                activeTab === 'all'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({templates.length})
            </button>
            <button
              onClick={() => setActiveTab('favorites')}
              className={`px-4 py-2 text-sm rounded-lg transition-colors flex items-center gap-2 ${
                activeTab === 'favorites'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Star className="w-4 h-4" />
              Favorites ({favorites.length})
            </button>
            <button
              onClick={() => {
                resetForm();
                setActiveTab('create');
              }}
              className={`px-4 py-2 text-sm rounded-lg transition-colors flex items-center gap-2 ${
                activeTab === 'create'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Plus className="w-4 h-4" />
              Create
            </button>
          </div>
          {activeTab !== 'create' && (
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search templates..."
                className="pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 w-64"
              />
            </div>
          )}
        </div>
      </div>

      {/* Notification */}
      {notification && (
        <div
          className={`mx-6 mt-4 p-3 rounded-lg flex items-center justify-between ${
            notification.type === 'success'
              ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
              : 'bg-red-500/10 border border-red-500/30 text-red-400'
          }`}
        >
          <span>{notification.message}</span>
          <button onClick={() => setNotification(null)}>
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === 'create' ? (
          /* Create Form */
          <div className={`${COMPONENTS.card.base} p-6 max-w-2xl`}>
            <h2 className="text-lg font-medium text-slate-100 mb-4">Create New Template</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={e => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="My prompt template"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={e => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Optional description..."
                  rows={2}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 resize-none"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-slate-300">
                    Prompts <span className="text-red-400">*</span>
                  </label>
                  <button
                    onClick={addPrompt}
                    className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" />
                    Add Prompt
                  </button>
                </div>
                <div className="space-y-2">
                  {formData.prompts.map((prompt, index) => (
                    <div key={index} className="flex gap-2">
                      <textarea
                        value={prompt}
                        onChange={e => updatePrompt(index, e.target.value)}
                        placeholder={`Prompt ${index + 1}...`}
                        rows={2}
                        className="flex-1 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 resize-none"
                      />
                      {formData.prompts.length > 1 && (
                        <button
                          onClick={() => removePrompt(index)}
                          className="p-2 text-slate-500 hover:text-red-400"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_favorite"
                  checked={formData.is_favorite}
                  onChange={e => setFormData(prev => ({ ...prev, is_favorite: e.target.checked }))}
                  className="rounded border-slate-700 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
                />
                <label htmlFor="is_favorite" className="text-sm text-slate-300">
                  Add to favorites
                </label>
              </div>
              <div className="flex justify-end gap-3 pt-4">
                <button
                  onClick={() => setActiveTab('all')}
                  className={`px-4 py-2 ${COMPONENTS.button.ghost}`}
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreate}
                  disabled={loading}
                  className={`px-4 py-2 ${COMPONENTS.button.primary}`}
                >
                  {loading ? 'Creating...' : 'Create Template'}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Template List */
          <div className="grid gap-4">
            {filteredTemplates.length === 0 ? (
              <div className={`${COMPONENTS.card.base} p-8 text-center`}>
                <FileText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-300 mb-2">
                  {searchQuery ? 'No matching templates' : 'No templates yet'}
                </h3>
                <p className="text-slate-500">
                  {searchQuery
                    ? 'Try a different search term'
                    : 'Create your first prompt template to get started'}
                </p>
              </div>
            ) : (
              filteredTemplates.map(template => (
                <div
                  key={template.id}
                  className={`${COMPONENTS.card.base} p-4 hover:border-slate-600 transition-colors`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium text-slate-100 truncate">{template.name}</h3>
                        {template.is_favorite && (
                          <Star className="w-4 h-4 text-amber-400 fill-amber-400 flex-shrink-0" />
                        )}
                      </div>
                      {template.description && (
                        <p className="text-sm text-slate-400 mb-2 line-clamp-2">
                          {template.description}
                        </p>
                      )}
                      <div className="text-xs text-slate-500">
                        {template.prompts.length} prompt{template.prompts.length !== 1 ? 's' : ''}
                        {template.tags && template.tags.length > 0 && (
                          <span className="ml-2">
                            {template.tags.map(tag => (
                              <span
                                key={tag}
                                className="inline-block px-2 py-0.5 bg-slate-800 rounded text-slate-400 mr-1"
                              >
                                {tag}
                              </span>
                            ))}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 ml-4">
                      <button
                        onClick={() => handleToggleFavorite(template.id)}
                        className={`p-2 rounded-lg transition-colors ${
                          template.is_favorite
                            ? 'text-amber-400 hover:bg-amber-500/10'
                            : 'text-slate-500 hover:text-amber-400 hover:bg-slate-800'
                        }`}
                        title={template.is_favorite ? 'Remove from favorites' : 'Add to favorites'}
                      >
                        <Star
                          className={`w-4 h-4 ${template.is_favorite ? 'fill-amber-400' : ''}`}
                        />
                      </button>
                      <button
                        onClick={() => handleDuplicate(template.id)}
                        className="p-2 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded-lg transition-colors"
                        title="Duplicate"
                      >
                        <Copy className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleEdit(template)}
                        className="p-2 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded-lg transition-colors"
                        title="Edit"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(template.id)}
                        className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Edit Modal */}
      {showEditModal && editingTemplate && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className={`${COMPONENTS.card.base} w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6 m-4`}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-slate-100">Edit Template</h2>
              <button
                onClick={() => {
                  setShowEditModal(false);
                  setEditingTemplate(null);
                }}
                className="text-slate-500 hover:text-slate-300"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={e => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={e => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  rows={2}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:border-emerald-500 resize-none"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-slate-300">Prompts</label>
                  <button
                    onClick={addPrompt}
                    className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" />
                    Add
                  </button>
                </div>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {formData.prompts.map((prompt, index) => (
                    <div key={index} className="flex gap-2">
                      <textarea
                        value={prompt}
                        onChange={e => updatePrompt(index, e.target.value)}
                        rows={2}
                        className="flex-1 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:border-emerald-500 resize-none"
                      />
                      {formData.prompts.length > 1 && (
                        <button
                          onClick={() => removePrompt(index)}
                          className="p-2 text-slate-500 hover:text-red-400"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="edit_is_favorite"
                  checked={formData.is_favorite}
                  onChange={e => setFormData(prev => ({ ...prev, is_favorite: e.target.checked }))}
                  className="rounded border-slate-700 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
                />
                <label htmlFor="edit_is_favorite" className="text-sm text-slate-300">
                  Favorite
                </label>
              </div>
              <div className="flex justify-end gap-3 pt-4">
                <button
                  onClick={() => {
                    setShowEditModal(false);
                    setEditingTemplate(null);
                  }}
                  className={`px-4 py-2 ${COMPONENTS.button.ghost}`}
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdate}
                  disabled={loading}
                  className={`px-4 py-2 ${COMPONENTS.button.primary}`}
                >
                  {loading ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
