/**
 * TrainingTemplateList component for displaying a list of training templates.
 *
 * This component renders a grid of template cards with search, filter, and pagination.
 */

import { useState } from 'react';
import { Search, SlidersHorizontal, Loader, ChevronLeft, ChevronRight } from 'lucide-react';
import { TrainingTemplate } from '../../types/trainingTemplate';
import { TrainingTemplateCard } from './TrainingTemplateCard';

interface TrainingTemplateListProps {
  templates: TrainingTemplate[];
  loading?: boolean;
  onTemplateClick?: (template: TrainingTemplate) => void;
  onEdit?: (template: TrainingTemplate) => void;
  onDelete?: (id: string) => void;
  onToggleFavorite?: (id: string) => void;
  onDuplicate?: (template: TrainingTemplate) => void;
  // Pagination
  currentPage?: number;
  totalPages?: number;
  hasNext?: boolean;
  hasPrev?: boolean;
  onPageChange?: (page: number) => void;
}

export function TrainingTemplateList({
  templates,
  loading = false,
  onTemplateClick,
  onEdit,
  onDelete,
  onToggleFavorite,
  onDuplicate,
  currentPage = 1,
  totalPages = 1,
  hasNext = false,
  hasPrev = false,
  onPageChange,
}: TrainingTemplateListProps) {
  const [localSearchQuery, setLocalSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Local filtering based on search query
  const filteredTemplates = templates.filter((template) => {
    if (!localSearchQuery) return true;
    const query = localSearchQuery.toLowerCase();
    return (
      template.name.toLowerCase().includes(query) ||
      (template.description && template.description.toLowerCase().includes(query)) ||
      template.encoder_type.toLowerCase().includes(query)
    );
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 text-slate-400 animate-spin" />
        <span className="ml-3 text-slate-400">Loading templates...</span>
      </div>
    );
  }

  if (templates.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 mb-4">
          <Search className="w-8 h-8 text-slate-500" />
        </div>
        <h3 className="text-lg font-semibold text-slate-300 mb-2">No templates found</h3>
        <p className="text-sm text-slate-500">
          Create your first training template to get started.
        </p>
      </div>
    );
  }

  if (filteredTemplates.length === 0 && localSearchQuery) {
    return (
      <div>
        <div className="mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              value={localSearchQuery}
              onChange={(e) => setLocalSearchQuery(e.target.value)}
              placeholder="Search templates..."
              className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            />
          </div>
        </div>
        <div className="text-center py-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 mb-4">
            <Search className="w-8 h-8 text-slate-500" />
          </div>
          <h3 className="text-lg font-semibold text-slate-300 mb-2">
            No templates match your search
          </h3>
          <p className="text-sm text-slate-500">
            Try adjusting your search terms or clear the search to see all templates.
          </p>
          <button
            onClick={() => setLocalSearchQuery('')}
            className="mt-4 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300"
          >
            Clear Search
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Search and Filters */}
      <div className="mb-6 flex gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={localSearchQuery}
            onChange={(e) => setLocalSearchQuery(e.target.value)}
            placeholder="Search templates by name, description, or architecture..."
            className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center gap-2 px-4 py-2 rounded transition-colors ${
            showFilters
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          <SlidersHorizontal className="w-4 h-4" />
          Filters
        </button>
      </div>

      {/* Filters Panel (placeholder for future enhancement) */}
      {showFilters && (
        <div className="mb-6 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
          <p className="text-sm text-slate-400">
            Additional filters coming soon (architecture type, date range, etc.)
          </p>
        </div>
      )}

      {/* Results Count */}
      <div className="mb-4 text-sm text-slate-400">
        Showing {filteredTemplates.length} of {templates.length} template
        {templates.length !== 1 ? 's' : ''}
        {localSearchQuery && ' (filtered)'}
      </div>

      {/* Template Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {filteredTemplates.map((template) => (
          <TrainingTemplateCard
            key={template.id}
            template={template}
            onClick={onTemplateClick ? () => onTemplateClick(template) : undefined}
            onEdit={onEdit}
            onDelete={onDelete}
            onToggleFavorite={onToggleFavorite}
            onDuplicate={onDuplicate}
          />
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && onPageChange && (
        <div className="flex items-center justify-between mt-6 pt-6 border-t border-slate-800">
          <div className="text-sm text-slate-400">
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={!hasPrev}
              className="flex items-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </button>
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={!hasNext}
              className="flex items-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded transition-colors"
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
