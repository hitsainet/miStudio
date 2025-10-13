/**
 * ExtractionTemplateList component for displaying a list of extraction templates.
 *
 * This component renders a grid of template cards with search, filter, and pagination.
 */

import React, { useState } from 'react';
import { Search, SlidersHorizontal, Loader } from 'lucide-react';
import { ExtractionTemplate } from '../../types/extractionTemplate';
import { ExtractionTemplateCard } from './ExtractionTemplateCard';

interface ExtractionTemplateListProps {
  templates: ExtractionTemplate[];
  loading?: boolean;
  onTemplateClick?: (template: ExtractionTemplate) => void;
  onEdit?: (template: ExtractionTemplate) => void;
  onDelete?: (id: string) => void;
  onToggleFavorite?: (id: string) => void;
  onDuplicate?: (template: ExtractionTemplate) => void;
  // Pagination
  currentPage?: number;
  totalPages?: number;
  hasNext?: boolean;
  hasPrev?: boolean;
  onPageChange?: (page: number) => void;
}

export function ExtractionTemplateList({
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
}: ExtractionTemplateListProps) {
  const [localSearchQuery, setLocalSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Local filtering based on search query
  const filteredTemplates = templates.filter((template) => {
    if (!localSearchQuery) return true;
    const query = localSearchQuery.toLowerCase();
    return (
      template.name.toLowerCase().includes(query) ||
      (template.description && template.description.toLowerCase().includes(query))
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
          Create your first extraction template to get started.
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
            Try adjusting your search query or filters.
          </p>
          <button
            onClick={() => setLocalSearchQuery('')}
            className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
          >
            Clear search
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={localSearchQuery}
            onChange={(e) => setLocalSearchQuery(e.target.value)}
            placeholder="Search templates by name or description..."
            className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`p-2 rounded transition-colors ${
            showFilters
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
          title="Toggle filters"
        >
          <SlidersHorizontal className="w-5 h-5" />
        </button>
      </div>

      {/* Filter Panel (placeholder for future expansion) */}
      {showFilters && (
        <div className="p-4 bg-slate-800 border border-slate-700 rounded">
          <p className="text-sm text-slate-400">
            Additional filters can be added here (sort order, hook types, etc.)
          </p>
        </div>
      )}

      {/* Results Count */}
      <div className="text-sm text-slate-400">
        Showing {filteredTemplates.length} of {templates.length} template
        {templates.length !== 1 ? 's' : ''}
      </div>

      {/* Template Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredTemplates.map((template) => (
          <ExtractionTemplateCard
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
        <div className="flex items-center justify-center gap-4 pt-6">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={!hasPrev}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded transition-colors"
          >
            Previous
          </button>
          <span className="text-sm text-slate-400">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={!hasNext}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
