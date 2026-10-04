import { useEffect } from 'react';
import { Cpu } from 'lucide-react';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

/** `"auto"`, `"all"`, or the UUID of the card a job should run on. */
export type GpuRequest = string;

export const AUTO_GPU: GpuRequest = 'auto';

/** Split the job's model across every GPU. Offered only where the job can run split. */
export const ALL_GPUS: GpuRequest = 'all';

interface GpuSelectProps {
  id: string;
  value: GpuRequest;
  onChange: (value: GpuRequest) => void;
  disabled?: boolean;
  /**
   * The job can run a model split across GPUs (multi-GPU Phase 2), so offer
   * "All GPUs". Off by default: a job that cannot split refuses `"all"`, and
   * offering a choice the backend refuses is a trap.
   */
  allowSplit?: boolean;
  label?: string;
  className?: string;
  /** Override the label classes so the picker matches its form's own labels. */
  labelClassName?: string;
  /** Override the select classes so the picker matches its form's own inputs. */
  selectClassName?: string;
}

const DEFAULT_LABEL_CLASS =
  'block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2';
const DEFAULT_SELECT_CLASS =
  'w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors';

/**
 * Which GPU a job runs on.
 *
 * "Auto" lets the backend take the card with the most free memory when the job
 * starts. Choosing a card sends its UUID rather than its index: adding a GPU
 * renumbered the existing card on 2026-09-13, and an index saved before that
 * would now name a different card.
 */
export function GpuSelect({
  id,
  value,
  onChange,
  disabled = false,
  allowSplit = false,
  label = 'GPU',
  className = '',
  labelClassName = DEFAULT_LABEL_CLASS,
  selectClassName = DEFAULT_SELECT_CLASS,
}: GpuSelectProps) {
  const { gpuList, fetchGPUList } = useSystemMonitorStore();

  useEffect(() => {
    if (!gpuList) {
      fetchGPUList();
    }
  }, [gpuList, fetchGPUList]);

  const gpus = gpuList?.gpus ?? [];
  // With one card there is nothing to split across.
  const offerSplit = allowSplit && gpus.length > 1;
  const isAll = value === ALL_GPUS;
  const missing = value !== AUTO_GPU && !isAll && !gpus.some((gpu) => gpu.uuid === value);

  return (
    <div className={className}>
      <label htmlFor={id} className={labelClassName}>
        <Cpu className="w-4 h-4 inline mr-1" />
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={selectClassName}
      >
        <option value={AUTO_GPU}>Auto — most free memory</option>
        {gpus.map((gpu) => (
          <option key={gpu.uuid} value={gpu.uuid}>
            GPU {gpu.gpu_id}: {gpu.name} ({Math.round(gpu.total_memory_gb)} GB)
          </option>
        ))}
        {offerSplit && <option value={ALL_GPUS}>All GPUs — split the model across cards</option>}
        {isAll && !offerSplit && (
          <option value={ALL_GPUS}>All GPUs (not supported for this job)</option>
        )}
        {missing && <option value={value}>Unavailable GPU ({value})</option>}
      </select>
    </div>
  );
}
