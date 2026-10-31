/**
 * Behaviour-detector reports store (Feature 040 S2; FTDD §7, 040-T39).
 *
 * Holds what the report surfaces show: the last breakdown per (probe, evaluation, column), success reports
 * per behaviour, and the model comparison. Every value is the server's; nothing is computed or invented
 * here. A refusal is kept as data (`refusal`: code, message, next) so a screen renders its `next` action
 * rather than a bare error string (FR-40).
 *
 * Each action calls exactly one client method and is called by a component
 * (`api/__tests__/probeClientIsReachable.test.ts` derives the pairs and checks both links).
 */
import { create } from 'zustand';

import { behaviourDetectorsApi } from '../api/behaviourDetectors';
import { APIError } from '../api/client';
import type {
  BreakdownQuery,
  ModelComparison,
  NextCall,
  ProbeBreakdown,
  RuleSetChoice,
  RuleSetValidation,
  RuleSignalsVerdict,
  RulesEvaluation,
  RulesEvaluationList,
  RulesEvaluationRequest,
  SuccessReportEnvelope,
  SuccessReportList,
  SuccessReportRequest,
} from '../types/probeMonitor';

export interface Refusal {
  code: string;
  message: string;
  next: NextCall[];
}

export function refusalOf(error: unknown): Refusal {
  if (error instanceof APIError && error.data) {
    return {
      code: String(error.data.code ?? 'refused'),
      message: String(error.data.message ?? error.detail),
      next: (error.data.next as NextCall[] | undefined) ?? [],
    };
  }
  return { code: 'error', message: error instanceof Error ? error.message : String(error), next: [] };
}

interface BehaviourDetectorsState {
  breakdown: ProbeBreakdown | null;
  report: SuccessReportEnvelope | null;
  reports: SuccessReportList | null;
  comparison: ModelComparison | null;
  busy: boolean;
  refusal: Refusal | null;

  fetchBreakdown: (probeId: string, evaluationId: string, query: BreakdownQuery) => Promise<void>;
  fetchSuccessReport: (key: string, body: SuccessReportRequest) => Promise<void>;
  readSuccessReport: (reportId: string) => Promise<void>;
  listSuccessReports: (key: string) => Promise<void>;
  fetchModelComparison: (key: string) => Promise<void>;
  updateViewColumns: (datasetId: string, groupColumns: string[]) => Promise<boolean>;
  // Feature 040 S3
  ruleValidation: RuleSetValidation | null;
  ruleVerdict: RuleSignalsVerdict | null;
  rulesEvaluation: RulesEvaluation | null;
  rulesEvaluations: RulesEvaluationList | null;
  validateRuleSet: (choice: RuleSetChoice) => Promise<void>;
  evaluateRuleSignals: (body: RuleSetChoice & { text?: string; scope?: string }) => Promise<void>;
  startRulesEvaluation: (probeId: string, body: RulesEvaluationRequest) => Promise<void>;
  pollRulesEvaluation: (evaluationId: string) => Promise<void>;
  listRulesEvaluations: (key: string) => Promise<void>;
  cancelRulesEvaluation: (evaluationId: string) => Promise<void>;
  // Feature 040 S4: the operator's revoke / re-approve of one operator code rule, then a fresh validation.
  revokeCodeRule: (codeSha256: string, reason: string, choice: RuleSetChoice) => Promise<void>;
  reapproveCodeRule: (codeSha256: string, reason: string, choice: RuleSetChoice) => Promise<void>;
}

export const useBehaviourDetectorsStore = create<BehaviourDetectorsState>()((set) => {
  async function run<T>(work: () => Promise<T>, apply: (value: T) => Partial<BehaviourDetectorsState>): Promise<boolean> {
    set({ busy: true, refusal: null });
    try {
      const value = await work();
      set({ ...apply(value), busy: false });
      return true;
    } catch (error) {
      set({ busy: false, refusal: refusalOf(error) });
      return false;
    }
  }

  return {
    breakdown: null,
    report: null,
    reports: null,
    comparison: null,
    busy: false,
    refusal: null,

    fetchBreakdown: async (probeId, evaluationId, query) => {
      await run(() => behaviourDetectorsApi.getBreakdown(probeId, evaluationId, query), (breakdown) => ({ breakdown }));
    },
    fetchSuccessReport: async (key, body) => {
      await run(() => behaviourDetectorsApi.createSuccessReport(key, body), (report) => ({ report }));
    },
    readSuccessReport: async (reportId) => {
      await run(() => behaviourDetectorsApi.readSuccessReport(reportId), (report) => ({ report }));
    },
    listSuccessReports: async (key) => {
      await run(() => behaviourDetectorsApi.listSuccessReports(key), (reports) => ({ reports }));
    },
    fetchModelComparison: async (key) => {
      await run(() => behaviourDetectorsApi.compareModels(key), (comparison) => ({ comparison }));
    },
    updateViewColumns: (datasetId, groupColumns) =>
      run(() => behaviourDetectorsApi.updateViewColumns(datasetId, groupColumns), () => ({})),

    // Feature 040 S3: rule sets and rules ∪ probe — every value is the server's.
    ruleValidation: null,
    ruleVerdict: null,
    rulesEvaluation: null,
    rulesEvaluations: null,
    validateRuleSet: async (choice) => {
      await run(() => behaviourDetectorsApi.validateRuleSet(choice), (ruleValidation) => ({ ruleValidation }));
    },
    evaluateRuleSignals: async (body) => {
      await run(() => behaviourDetectorsApi.evaluateRuleSignals(body), (ruleVerdict) => ({ ruleVerdict }));
    },
    startRulesEvaluation: async (probeId, body) => {
      await run(() => behaviourDetectorsApi.startRulesEvaluation(probeId, body), (rulesEvaluation) => ({ rulesEvaluation }));
    },
    pollRulesEvaluation: async (evaluationId) => {
      await run(() => behaviourDetectorsApi.getRulesEvaluation(evaluationId), (rulesEvaluation) => ({ rulesEvaluation }));
    },
    listRulesEvaluations: async (key) => {
      await run(() => behaviourDetectorsApi.listRulesEvaluations(key), (rulesEvaluations) => ({ rulesEvaluations }));
    },
    cancelRulesEvaluation: async (evaluationId) => {
      await run(() => behaviourDetectorsApi.cancelRulesEvaluation(evaluationId), () => ({}));
    },
    revokeCodeRule: async (codeSha256, reason, choice) => {
      if (await run(() => behaviourDetectorsApi.revokeCodeRule(codeSha256, reason), () => ({}))) {
        await run(() => behaviourDetectorsApi.validateRuleSet(choice), (ruleValidation) => ({ ruleValidation }));
      }
    },
    reapproveCodeRule: async (codeSha256, reason, choice) => {
      if (await run(() => behaviourDetectorsApi.reapproveCodeRule(codeSha256, reason), () => ({}))) {
        await run(() => behaviourDetectorsApi.validateRuleSet(choice), (ruleValidation) => ({ ruleValidation }));
      }
    },
  };
});
