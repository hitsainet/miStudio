/**
 * Band shading on the trajectory chart.
 *
 * This is the only test that exercises band rendering at all: no band report
 * exists in the product yet (doc chain 021 produces the first one), so the path
 * is unreachable from the panel. Testing it HERE, at its real caller, is the
 * difference between "implemented" and "implemented and known to work" — and
 * the code is written to be dead until a report exists precisely because BR-002
 * forbids inventing one.
 *
 * MUTATION CONTROLS:
 *   * drop type="number" from the XAxis   -> the band rect lands wrong / vanishes
 *   * render the ReferenceArea when bandReport is null -> "no bands" fails
 */

import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { TrajectoryChart } from './TrajectoryChart';
import type { BandReport, LensTypeSlice } from '../../types/jlens';

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

/**
 * DELIBERATELY NON-UNIFORM.
 *
 * On an evenly spaced 0,10,...,100 axis a category axis and a numeric axis put
 * layer 90 in the same place, so the band lands correctly either way and the
 * assertion proves nothing. Real axes are not guaranteed uniform, and this one
 * separates the two: layer 90 sits at 90% of the numeric range but at 6/7 of
 * the category range, and layer 40 is not a category at all.
 */
const AXIS = [0, 5, 10, 15, 20, 80, 90, 100];

const SLICE: LensTypeSlice = {
  type: 'LOGIT_LENS',
  top_tokens: AXIS.map(() => ['alpha', 'beta']),
  top_probs: AXIS.map(() => [0.6, 0.2]),
};

const REPORT: BandReport = {
  model: 'test-model',
  workspace_start: 40,
  motor_start: 90,
  derivation: 'computed for this model',
};

function renderChart(bandReport: BandReport | null) {
  return render(
    <TrajectoryChart
      axis={AXIS}
      slice={SLICE}
      pinned={['alpha']}
      topN={2}
      selPos={0}
      bandReport={bandReport}
    />
  );
}

describe('band shading', () => {
  it('draws nothing when there is no report', () => {
    const { container } = renderChart(null);
    expect(container.querySelectorAll('.recharts-reference-area').length).toBe(0);
  });

  it('draws the band from the report, positioned by layer NUMBER', () => {
    const { container } = renderChart(REPORT);
    // recharts renders a Rectangle as a <path>, not a <rect>.
    const band = container.querySelector('.recharts-reference-area-rect');
    expect(band).not.toBeNull();

    // "M x,y h w ..." — the band spans layers 40..90 of a 0..100 axis, so half
    // the plot width. On a CATEGORY axis the numbers 40 and 90 are matched
    // against category labels rather than coordinates, and the band lands in
    // the wrong place or collapses.
    const d = band!.getAttribute('d') ?? '';
    const [, xs, , ws] = d.match(/^M\s*([\d.]+),\s*([\d.]+)\s*h\s*([\d.-]+)/) ?? [];
    const x = Number(xs);
    const width = Number(ws);

    const axisLine = container.querySelector('.recharts-xAxis .recharts-cartesian-axis-line');
    const plotWidth = Number(axisLine?.getAttribute('width') ?? 0);
    const plotX = Number(axisLine?.getAttribute('x') ?? 0);

    expect(plotWidth).toBeGreaterThan(0);
    // Layers 40..90 of a 0..100 range: starts at 40% of the plot, spans 50%.
    expect((x - plotX) / plotWidth).toBeCloseTo(0.4, 1);
    expect(width / plotWidth).toBeCloseTo(0.5, 1);
  });
});
