/**
 * Expandable list of feature groups with search/sort controls.
 */

import { ChevronDown, ChevronRight, Search } from 'lucide-react';
import { useFeatureGroupsStore } from '../../stores/featureGroupsStore';
import { cleanDisplayText } from '../../utils/tokenDisplay';
import { GroupMembersTable } from './GroupMembersTable';

interface GroupListProps {
  onOpenFeature: (featureId: string) => void;
}

export function GroupList({ onOpenFeature }: GroupListProps) {
  const {
    groups,
    groupsTotal,
    groupsOffset,
    filters,
    setFilters,
    fetchGroups,
    expandedGroupId,
    expandGroup,
    isLoading,
  } = useFeatureGroupsStore();

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex flex-wrap items-center gap-3 mb-3 shrink-0">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <input
            value={filters.search}
            onChange={(e) => setFilters({ search: e.target.value })}
            placeholder="Search token…"
            className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-md pl-8 pr-3 py-1.5 text-sm text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-600 focus:border-emerald-500 focus:outline-none w-48"
          />
        </div>
        <label className="text-xs text-slate-500 flex items-center gap-1.5">
          Min size
          <select
            value={filters.minGroupSize}
            onChange={(e) => setFilters({ minGroupSize: Number(e.target.value) })}
            className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-md px-2 py-1 text-sm text-slate-800 dark:text-slate-200"
          >
            {[2, 3, 5, 10].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500 flex items-center gap-1.5">
          Sort
          <select
            value={filters.sortBy}
            onChange={(e) => setFilters({ sortBy: e.target.value as 'size' | 'cohesion' | 'token' })}
            className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-md px-2 py-1 text-sm text-slate-800 dark:text-slate-200"
          >
            <option value="size">Size</option>
            <option value="cohesion">Cohesion</option>
            <option value="token">Token</option>
          </select>
        </label>
        <span className="text-xs text-slate-600 ml-auto">
          {groupsTotal.toLocaleString()} clusters
        </span>
      </div>

      {/* Only this region scrolls; the filter bar above stays pinned. */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-1 pb-2">
      {isLoading && groups.length === 0 && (
        <p className="text-sm text-slate-500 py-6 text-center">Loading clusters…</p>
      )}
      {!isLoading && groups.length === 0 && (
        <p className="text-sm text-slate-500 py-6 text-center">
          No clusters match the current filters.
        </p>
      )}

      <div className="space-y-1">
        {groups.map((group) => {
          const expanded = expandedGroupId === group.group_id;
          return (
            <div key={group.group_id} className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
              <button
                onClick={() => void expandGroup(group.group_id)}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800/50 rounded-lg"
              >
                {expanded ? (
                  <ChevronDown className="w-4 h-4 text-slate-500 shrink-0" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-slate-500 shrink-0" />
                )}
                <span className="font-mono text-emerald-400 text-sm">
                  "{cleanDisplayText(group.display_token)}"
                </span>
                <span className="text-xs text-slate-500">
                  {group.member_count} members · cohesion {group.cohesion.toFixed(2)}
                </span>
                {group.sample_labels.length > 0 && (
                  <span className="text-xs text-slate-600 truncate hidden md:inline">
                    {group.sample_labels.join(' · ')}
                  </span>
                )}
              </button>
              {expanded && <GroupMembersTable onOpenFeature={onOpenFeature} />}
            </div>
          );
        })}
      </div>

      {groupsTotal > groups.length + groupsOffset || groupsOffset > 0 ? (
        <div className="flex items-center justify-center gap-3 mt-4">
          <button
            disabled={groupsOffset === 0}
            onClick={() => void fetchGroups(Math.max(0, groupsOffset - 50))}
            className="px-3 py-1 text-xs bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-xs text-slate-500">
            {groupsOffset + 1}–{Math.min(groupsOffset + 50, groupsTotal)} of {groupsTotal}
          </span>
          <button
            disabled={groupsOffset + 50 >= groupsTotal}
            onClick={() => void fetchGroups(groupsOffset + 50)}
            className="px-3 py-1 text-xs bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded disabled:opacity-40"
          >
            Next
          </button>
        </div>
      ) : null}
      </div>
    </div>
  );
}
