/**
 * Recharts 2 -> 3 (2026-10-21): two silent changes in the monitor's utilization chart, found by
 * the before/after screenshots, not by any test (nothing rendered this chart):
 *   * the legend is sorted alphabetically by default, so it read "CPU Util, GPU Temp, GPU Util"
 *     instead of the order the lines are drawn in;
 *   * the grid's horizontal lines follow the y axis whose id it names; this chart's axes are
 *     "left" and "right", the grid's default id is 0, and the lines disappeared.
 *
 * MUTATION CONTROLS: drop `itemSorter` -> the legend test fails; drop the grid's `yAxisId` ->
 * the grid test fails.
 */
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { UtilizationChart } from './UtilizationChart';

vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <actual.ResponsiveContainer width={800} height={300}>
        {children}
      </actual.ResponsiveContainer>
    ),
  };
});

const DATA = [0, 1, 2, 3].map((i) => ({
  timestamp: 1_790_000_000_000 + i * 1000,
  gpu_utilization: 40 + i,
  cpu_utilization: 20 + i,
  gpu_temperature: 60 + i,
}));

describe('UtilizationChart', () => {
  it('lists the legend in the order the series are drawn', () => {
    const { container } = render(<UtilizationChart data={DATA} />);
    const labels = Array.from(container.querySelectorAll('.recharts-legend-item-text')).map(
      (el) => el.textContent
    );
    expect(labels).toEqual(['GPU Util', 'CPU Util', 'GPU Temp']);
  });

  it('draws the horizontal grid lines', () => {
    const { container } = render(<UtilizationChart data={DATA} />);
    expect(container.querySelector('.recharts-cartesian-grid')).not.toBeNull();
    // Without the axis binding Recharts 3 still draws the two border lines (top and bottom);
    // the 0/25/50/75/100 tick lines are what goes missing.
    expect(container.querySelectorAll('.recharts-cartesian-grid-horizontal line').length).toBeGreaterThan(2);
  });
});
