/**
 * GPUSelector Component
 *
 * Dropdown for selecting which GPU to monitor in single view mode
 */

import { Monitor } from 'lucide-react';
import { GPUInfo } from '../../types/system';

interface GPUSelectorProps {
  gpus: GPUInfo[];
  selected: number;
  onChange: (gpuId: number) => void;
}

export function GPUSelector({ gpus, selected, onChange }: GPUSelectorProps) {
  if (gpus.length === 0) {
    return null;
  }

  // Don't show dropdown if only one GPU
  if (gpus.length === 1) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <Monitor className="w-4 h-4" />
        <span>{gpus[0].name}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Monitor className="w-4 h-4 text-slate-400" />
      <select
        value={selected}
        onChange={(e) => onChange(Number(e.target.value))}
        className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
      >
        {gpus.map((gpu) => (
          <option key={gpu.gpu_id} value={gpu.gpu_id}>
            GPU {gpu.gpu_id}: {gpu.name} ({gpu.total_memory_gb}GB)
          </option>
        ))}
      </select>
    </div>
  );
}
