// @vitest-environment-options {"url": "http://k8s-mistudio.hitsai.local/"}
/**
 * The address-moved banner is retired (ENH-009 042 FR-042.35; FTASKS 18.11, group G9; control K-18.G9).
 *
 * From P3 to P7 the app shell showed "This address has moved" on an OLD browser host (k8s-mistudio.*), with the
 * count of drafts the browser kept for the old origin. Since P7 the Ingress serves no old host (group G5), so the
 * banner, `config/hosts.ts` and its date are gone. The real <App/> renders here on the old host itself (the jsdom
 * URL above) with a draft saved, which is exactly when the banner used to show: restoring its mount turns this red.
 */
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import App from '../../App';

vi.mock('../panels/DatasetsPanel', () => ({ DatasetsPanel: () => <h1>Datasets stub</h1> }));
vi.mock('../panels/ModelsPanel', () => ({ ModelsPanel: () => <div /> }));
vi.mock('../panels/TemplatesPanel', () => ({ TemplatesPanel: () => <div /> }));
vi.mock('../panels/TrainingPanel', () => ({ TrainingPanel: () => <h1>Training stub</h1> }));
vi.mock('../panels/ExtractionsPanel', () => ({ ExtractionsPanel: () => <div /> }));
vi.mock('../panels/LabelingPanel', () => ({ LabelingPanel: () => <div /> }));
vi.mock('../panels/CircuitsPanel', () => ({ CircuitsPanel: () => <div /> }));
vi.mock('../panels/ProbeMonitorsPanel', () => ({ ProbeMonitorsPanel: () => <div /> }));
vi.mock('../panels/JLensPanel', () => ({ JLensPanel: () => <div /> }));
vi.mock('../panels/SAEsPanel', () => ({ SAEsPanel: () => <div /> }));
vi.mock('../panels/SteeringPanel', () => ({ SteeringPanel: () => <div /> }));
vi.mock('../panels/FeatureGroupsPanel', () => ({ FeatureGroupsPanel: () => <div /> }));
vi.mock('../panels/SettingsPanel', () => ({ SettingsPanel: () => <div /> }));
vi.mock('../SystemMonitor/SystemMonitor', () => ({ SystemMonitor: () => <div /> }));
vi.mock('../SystemMonitor/CompactGPUStatus', () => ({ CompactGPUStatus: () => null }));
vi.mock('./ApprovalsBanner', () => ({ ApprovalsBanner: () => null }));
vi.mock('../../hooks/useDatasetProgress', () => ({ useGlobalDatasetProgress: () => {} }));

describe('AddressMovedBanner', () => {
  it('banner is retired', () => {
    expect(window.location.hostname).toBe('k8s-mistudio.hitsai.local');
    localStorage.clear();
    sessionStorage.clear();
    localStorage.setItem('mistudio.draft.probe-run', '{"a":1}');
    render(<App />);
    expect(screen.getByRole('heading', { name: 'Datasets stub' })).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'This address has moved' })).toBeNull();
    expect(screen.queryByText(/This address keeps working/)).toBeNull();
    // The modules went with the mount: nothing left to mount by accident.
    expect(Object.keys(import.meta.glob('./AddressMovedBanner.tsx'))).toEqual([]);
    expect(Object.keys(import.meta.glob('../../config/hosts.ts'))).toEqual([]);
  });
});
