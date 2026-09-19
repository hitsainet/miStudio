/**
 * Members of an expanded feature group: selection, labels, snippets,
 * click-through to feature detail, related-features lookup.
 */

import { useEffect, useRef } from 'react';
import { Link2, Star } from 'lucide-react';
import { useFeatureGroupsStore } from '../../stores/featureGroupsStore';
import { cleanDisplayText } from '../../utils/tokenDisplay';
import type { FeatureGroupMember } from '../../types/featureGroups';

interface GroupMembersTableProps {
  onOpenFeature: (featureId: string) => void;
}

const STAR_COLORS: Record<string, string> = {
  yellow: 'text-yellow-400',
  purple: 'text-purple-400',
  aqua: 'text-cyan-400',
};

export function GroupMembersTable({ onOpenFeature }: GroupMembersTableProps) {
  const { groupDetail, selection, toggleSelect, setSelected, fetchRelated } =
    useFeatureGroupsStore();
  const selectAllRef = useRef<HTMLInputElement>(null);

  const selectedCount = groupDetail
    ? groupDetail.members.filter((m) => selection.has(m.feature_id)).length
    : 0;
  const allSelected = groupDetail ? selectedCount === groupDetail.members.length : false;
  const someSelected = selectedCount > 0 && !allSelected;

  // Native indeterminate state can only be set imperatively.
  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = someSelected;
  }, [someSelected]);

  if (!groupDetail) {
    return <div className="text-xs text-slate-500 py-3 pl-8">Loading members…</div>;
  }

  return (
    <div className="pl-8 pr-2 pb-3">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-slate-500 text-left">
            <th className="w-8">
              <input
                ref={selectAllRef}
                type="checkbox"
                checked={allSelected}
                onChange={() =>
                  setSelected(
                    groupDetail.members.map((m) => ({
                      feature_id: m.feature_id,
                      neuron_index: m.neuron_index,
                      max_activation: m.max_activation,
                      activation_frequency: m.activation_frequency,
                      group_id: groupDetail.group_id,
                      display_token: groupDetail.display_token,
                      similarity: m.similarity,
                      cohesion: groupDetail.cohesion,
                    })),
                    !allSelected,
                  )
                }
                className="accent-emerald-500"
                aria-label={allSelected ? 'Deselect all members' : 'Select all members'}
                title={allSelected ? 'Select none' : 'Select all'}
              />
            </th>
            <th className="py-1 pr-3 w-16">#</th>
            <th className="py-1 pr-3">Label</th>
            <th className="py-1 pr-3 hidden lg:table-cell">Context</th>
            <th className="py-1 pr-3 w-20 text-right">Sim</th>
            <th className="py-1 w-16"></th>
          </tr>
        </thead>
        <tbody>
          {groupDetail.members.map((member: FeatureGroupMember) => (
            <tr
              key={member.feature_id}
              className="border-t border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800/40"
            >
              <td className="py-1.5">
                <input
                  type="checkbox"
                  checked={selection.has(member.feature_id)}
                  onChange={() =>
                    toggleSelect(member.feature_id, {
                      neuron_index: member.neuron_index,
                      max_activation: member.max_activation,
                      activation_frequency: member.activation_frequency,
                      group_id: groupDetail.group_id,
                      display_token: groupDetail.display_token,
                      similarity: member.similarity,
                      cohesion: groupDetail.cohesion,
                    })
                  }
                  className="accent-emerald-500"
                  aria-label={`Select feature ${member.neuron_index}`}
                />
              </td>
              <td className="py-1.5 pr-3 text-slate-600 dark:text-slate-400 font-mono text-xs">
                {member.neuron_index}
              </td>
              <td className="py-1.5 pr-3">
                <button
                  onClick={() => onOpenFeature(member.feature_id)}
                  className="text-left text-slate-800 dark:text-slate-200 hover:text-emerald-400 flex items-center gap-1.5"
                >
                  {member.star_color && (
                    <Star
                      className={`w-3 h-3 shrink-0 fill-current ${STAR_COLORS[member.star_color] ?? 'text-slate-500'}`}
                    />
                  )}
                  <span className={member.label_source === 'auto' ? 'text-slate-500 italic' : ''}>
                    {member.name}
                  </span>
                </button>
              </td>
              <td className="py-1.5 pr-3 hidden lg:table-cell">
                <span className="text-xs text-slate-500">
                  {cleanDisplayText(member.context_snippet)}
                </span>
              </td>
              <td className="py-1.5 pr-3 text-right text-xs text-slate-600 dark:text-slate-400">
                {member.similarity.toFixed(2)}
              </td>
              <td className="py-1.5 text-right">
                <button
                  onClick={() => void fetchRelated(member.feature_id)}
                  className="text-slate-500 hover:text-emerald-400"
                  title="Find related features"
                >
                  <Link2 className="w-3.5 h-3.5" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
