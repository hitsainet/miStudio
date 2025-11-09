# Test Coverage Expansion

**Created:** 2025-11-09
**Priority:** P0 - Blocks Production
**Current Coverage:** Unit 40%, Integration 20%
**Target Coverage:** Unit 70%, Integration 50%
**Estimated Time:** 84-108 hours (10-14 developer days)
**Status:** PLANNED - Ready for Execution

---

## Executive Summary

Comprehensive testing initiative to expand test coverage from current levels (40% unit, 20% integration) to production-ready targets (70% unit, 50% integration). This work is critical for production deployment and addresses gaps identified in the code review conducted on 2025-11-09.

### Why This Matters
- **Production Readiness:** Cannot deploy with 40% test coverage - too high risk of regressions
- **Code Review Findings:** 52 failing tests need investigation, WebSocket reliability untested
- **Recent Features Untested:** Multi-tokenization, feature labeling separation, dual-label system
- **Confidence:** High test coverage enables confident refactoring and feature additions

### Coverage Goals
| Test Type | Current | Target | Gap |
|-----------|---------|--------|-----|
| **Unit Tests** | 40% | 70% | +30 pp |
| **Integration Tests** | 20% | 50% | +30 pp |
| **E2E Tests** | 10% | 30% | +20 pp |

---

## Phase 1: Fix Failing Tests (12-16 hours) 🔴 CRITICAL

**Priority:** P0 - Must complete before any other phases
**Status:** NOT STARTED

### Overview
Address 52 failing tests identified in code review. These tests may be failing due to:
- Deprecated fixtures or mocks
- Changes in API contracts
- Database schema changes (e.g., multi-tokenization migration)
- Outdated assertions

### Tasks

- [ ] **1.1 Investigate all 52 failing tests** (4-6 hours)
  - Run full test suite: `pytest` (backend), `npm test` (frontend)
  - Categorize failures by type:
    - Deprecated/outdated tests (safe to remove)
    - Tests needing fixture updates
    - Tests failing due to schema changes
    - Tests with incorrect assertions
  - Document findings in investigation spreadsheet

- [ ] **1.2 Fix or remove deprecated tests** (3-4 hours)
  - Remove tests for features that no longer exist
  - Update tests for refactored code
  - Migrate deprecated assertion patterns to current standards

- [ ] **1.3 Update test fixtures** (3-4 hours)
  - Update database fixtures for multi-tokenization schema
  - Update model fixtures for any new fields (e.g., `has_completed_extractions`)
  - Update training fixtures for any new fields

- [ ] **1.4 Verify all tests passing** (2 hours)
  - Run full backend test suite: `pytest --verbose`
  - Run full frontend test suite: `npm test`
  - Achieve 100% pass rate before proceeding
  - Document any tests that cannot be fixed (mark as skipped with reason)

**Acceptance Criteria:**
- ✅ All 52 failing tests investigated and categorized
- ✅ Failing tests either fixed or removed with justification
- ✅ Test suite achieves 100% pass rate (or acceptable skip rate with documented reasons)
- ✅ Investigation findings documented for future reference

---

## Phase 2: WebSocket Reliability Tests (16-20 hours) 🔴 CRITICAL

**Priority:** P0 - WebSocket is core to real-time updates
**Status:** NOT STARTED
**Dependencies:** Phase 1 complete (all tests passing)

### Overview
WebSocket reliability is untested but critical for production. Need comprehensive tests covering connection lifecycle, reconnection, message ordering, and concurrent connections.

### Tasks

- [ ] **2.1 Test WebSocket connection/disconnection** (4-5 hours)
  - Test: Client connects to WebSocket server successfully
  - Test: Client disconnects gracefully on unmount
  - Test: Server handles client disconnect without errors
  - Test: Connection state tracked correctly (connected/disconnected)
  - **File:** `frontend/src/hooks/useWebSocket.test.ts` (NEW)

- [ ] **2.2 Test reconnection and state synchronization** (4-5 hours)
  - Test: Client reconnects automatically after disconnect
  - Test: Subscriptions restored after reconnection
  - Test: State synchronized after reconnection (fetch latest data)
  - Test: Exponential backoff on repeated reconnection failures
  - **File:** `frontend/src/hooks/useWebSocket.reconnection.test.ts` (NEW)

- [ ] **2.3 Test message ordering and delivery** (3-4 hours)
  - Test: Messages delivered in order (no out-of-order delivery)
  - Test: No duplicate messages on reconnection
  - Test: No lost messages during high-frequency updates
  - Test: Message delivery acknowledged correctly
  - **File:** `backend/tests/integration/test_websocket_message_ordering.py` (NEW)

- [ ] **2.4 Test concurrent connections** (4-5 hours)
  - Test: Multiple clients can connect simultaneously
  - Test: Messages delivered to correct clients (channel isolation)
  - Test: No cross-contamination between channels
  - Test: Server handles 100+ concurrent connections
  - **File:** `backend/tests/integration/test_websocket_concurrent_connections.py` (NEW)

- [ ] **2.5 Test error handling** (1-2 hours)
  - Test: Invalid channel subscription returns error
  - Test: Malformed message handled gracefully
  - Test: Server continues functioning after client error
  - **File:** `backend/tests/unit/test_websocket_error_handling.py` (NEW)

**Acceptance Criteria:**
- ✅ WebSocket connection lifecycle fully tested (connect, disconnect, reconnect)
- ✅ Message ordering and delivery verified under various conditions
- ✅ Concurrent connections tested with realistic load (100+ clients)
- ✅ Error handling covers all identified failure modes
- ✅ Integration tests cover full WebSocket emission flows (backend → frontend)

---

## Phase 3: Multi-Tokenization Tests (12-16 hours) 🔴 CRITICAL

**Priority:** P0 - New feature completely untested
**Status:** NOT STARTED
**Dependencies:** Phase 1 complete

### Overview
Multi-tokenization architecture implemented in 2025-11-09 session is completely untested. Need comprehensive tests for DatasetTokenization model, CRUD operations, cancellation, and migration scripts.

### Tasks

- [ ] **3.1 Test DatasetTokenization model CRUD** (3-4 hours)
  - Test: Create tokenization record with valid data
  - Test: Read tokenization by dataset_id and model_id
  - Test: Update tokenization progress and status
  - Test: Delete tokenization record
  - Test: Unique constraint on (dataset_id, model_id) enforced
  - **File:** `backend/tests/unit/test_dataset_tokenization_model.py` (NEW)

- [ ] **3.2 Test multiple tokenizations per dataset** (3-4 hours)
  - Test: Create multiple tokenizations for same dataset (different models)
  - Test: Fetch all tokenizations for a dataset
  - Test: Statistics isolated per tokenization (no cross-contamination)
  - Test: Delete one tokenization doesn't affect others
  - **File:** `backend/tests/integration/test_multi_tokenization_workflow.py` (NEW)

- [ ] **3.3 Test tokenization cancellation** (2-3 hours)
  - Test: Cancel running tokenization job (PROCESSING status)
  - Test: Cancel queued tokenization job (QUEUED status)
  - Test: Cannot cancel completed tokenization (ERROR status)
  - Test: Celery task revoked correctly on cancellation
  - Test: Status updated to CANCELLED in database
  - **File:** `backend/tests/integration/test_tokenization_cancellation.py` (NEW)

- [ ] **3.4 Test migration scripts** (4-5 hours)
  - Test: Migration 04b58ed9486a creates dataset_tokenizations table
  - Test: Migration 2e1feb9cc451 migrates existing data correctly
  - Test: Migration 7282abcac53a removes legacy fields
  - Test: Rollback migrations work correctly (downgrade path)
  - Test: Data integrity maintained throughout migration
  - **File:** `backend/tests/integration/test_tokenization_migrations.py` (NEW)

**Acceptance Criteria:**
- ✅ DatasetTokenization model CRUD operations fully tested
- ✅ Multiple tokenizations per dataset verified working correctly
- ✅ Cancellation functionality tested for all valid status transitions
- ✅ Migration scripts tested with realistic data scenarios
- ✅ Test coverage for multi-tokenization ≥70%

---

## Phase 4: Feature Labeling Tests (8-12 hours) 🟡 HIGH PRIORITY

**Priority:** P1 - Feature labeling separation implemented but untested
**Status:** NOT STARTED
**Dependencies:** Phase 1 complete

### Overview
Dual-label system (GPT labels vs manual labels) and feature labeling separation implemented but completely untested. Need tests for label separation logic, labeling job management, and dual-label system.

### Tasks

- [ ] **4.1 Test dual-label system** (3-4 hours)
  - Test: Features have two label fields (gpt_label, manual_label)
  - Test: GPT labels automatically generated via labeling job
  - Test: Manual labels override GPT labels in display
  - Test: Label priority logic (manual > GPT > null)
  - **File:** `backend/tests/unit/test_dual_label_system.py` (NEW)

- [ ] **4.2 Test label separation logic** (2-3 hours)
  - Test: Feature extraction doesn't trigger labeling
  - Test: Labeling job runs independently of extraction
  - Test: Labels stored correctly in database
  - Test: Labels fetched correctly in API responses
  - **File:** `backend/tests/unit/test_label_separation_logic.py` (NEW)

- [ ] **4.3 Test labeling job management** (3-5 hours)
  - Test: Create labeling job for extracted features
  - Test: Labeling job processes features in batches
  - Test: Progress tracked correctly (features labeled / total features)
  - Test: Cancellation of labeling jobs
  - Test: Error handling for GPT API failures
  - **File:** `backend/tests/integration/test_labeling_job_workflow.py` (NEW)

**Acceptance Criteria:**
- ✅ Dual-label system fully tested (GPT + manual labels)
- ✅ Label separation logic verified (extraction → labeling independence)
- ✅ Labeling job lifecycle tested (create, progress, complete, cancel)
- ✅ Test coverage for feature labeling ≥60%

---

## Phase 5: Unit Test Expansion (20-24 hours) 🟡 HIGH PRIORITY

**Priority:** P1 - Increase unit coverage from 40% to 70%
**Status:** NOT STARTED
**Dependencies:** Phase 1 complete

### Overview
Expand unit test coverage to 70% by focusing on untested or under-tested areas: progress calculation functions, error recovery flows, state management stores, API client functions.

### Tasks

- [ ] **5.1 Test progress calculation functions** (6-8 hours)
  - Test: Training progress calculation (step / total_steps * 100)
  - Test: Extraction progress calculation (loading, extracting, saving phases)
  - Test: Tokenization progress calculation (0%, 10%, 20%, 40%, 80%, 95%, 100%)
  - Test: Progress clamping (0-100 bounds)
  - Test: Edge cases (zero total, negative values, overflow)
  - **Files:**
    - `backend/tests/unit/test_training_progress.py` (EXPAND)
    - `backend/tests/unit/test_extraction_progress.py` (EXPAND)
    - `backend/tests/unit/test_tokenization_progress.py` (NEW)

- [ ] **5.2 Test error recovery flows** (5-6 hours)
  - Test: OOM error recovery (batch size reduction)
  - Test: Network error retry with exponential backoff
  - Test: Database connection failure recovery
  - Test: Celery task retry logic (transient vs permanent errors)
  - Test: Error state persistence in database
  - **Files:**
    - `backend/tests/unit/test_error_recovery.py` (NEW)
    - `backend/tests/unit/test_celery_retry_logic.py` (NEW)

- [ ] **5.3 Test state management stores** (5-6 hours)
  - Test: trainingsStore actions (create, update, delete, pause, resume, stop)
  - Test: datasetsStore tokenization methods (fetch, cancel, delete)
  - Test: modelsStore extraction methods (start, cancel, fetch progress)
  - Test: WebSocket update handlers in stores
  - Test: Store state consistency after concurrent updates
  - **Files:**
    - `frontend/src/stores/trainingsStore.test.ts` (EXPAND)
    - `frontend/src/stores/datasetsStore.test.ts` (EXPAND)
    - `frontend/src/stores/modelsStore.test.ts` (EXPAND)

- [ ] **5.4 Test API client functions** (4-5 hours)
  - Test: API client request/response handling
  - Test: Error handling for 400, 404, 500 status codes
  - Test: Request retries for transient failures
  - Test: Authentication token injection
  - Test: Query parameter serialization
  - **Files:**
    - `frontend/src/api/trainings.test.ts` (NEW)
    - `frontend/src/api/datasets.test.ts` (EXPAND)
    - `frontend/src/api/models.test.ts` (NEW)

**Acceptance Criteria:**
- ✅ Unit test coverage increased from 40% to ≥70%
- ✅ All progress calculation functions tested with edge cases
- ✅ Error recovery flows verified for all major error types
- ✅ State management stores tested for all CRUD operations
- ✅ API client functions tested for success and error cases

---

## Phase 6: Integration Test Expansion (16-20 hours) 🟡 HIGH PRIORITY

**Priority:** P1 - Increase integration coverage from 20% to 50%
**Status:** NOT STARTED
**Dependencies:** Phase 1 complete, Phases 2-5 recommended

### Overview
Expand integration test coverage to 50% by testing end-to-end flows: WebSocket emission, training job lifecycle, dataset tokenization, feature extraction.

### Tasks

- [ ] **6.1 Test WebSocket emission flows** (4-5 hours)
  - Test: Training progress emission (backend → WebSocket → frontend)
  - Test: Extraction progress emission
  - Test: Tokenization progress emission
  - Test: System monitoring metrics emission
  - Test: Message delivery verified in frontend stores
  - **File:** `backend/tests/integration/test_websocket_emission_integration.py` (EXPAND)

- [ ] **6.2 Test training job lifecycle** (5-6 hours)
  - Test: Create training → queue → start → progress updates → complete
  - Test: Create training → pause → resume → complete
  - Test: Create training → stop (manual termination)
  - Test: Create training → OOM error → batch size reduction → continue
  - Test: Checkpoint creation at intervals
  - Test: Metrics saved to database throughout lifecycle
  - **File:** `backend/tests/integration/test_training_lifecycle_e2e.py` (NEW)

- [ ] **6.3 Test dataset tokenization flow** (3-4 hours)
  - Test: Create tokenization → queue → process → save statistics → complete
  - Test: Create multiple tokenizations for same dataset (different models)
  - Test: Cancel tokenization mid-process
  - Test: Error handling for invalid tokenizer
  - **File:** `backend/tests/integration/test_tokenization_flow_e2e.py` (NEW)

- [ ] **6.4 Test feature extraction flow** (4-5 hours)
  - Test: Start extraction → load model → extract activations → save to disk → complete
  - Test: Progress updates emitted correctly (0%, 10%, 50%, 90%, 100%)
  - Test: Extraction failure handling (OOM, model load failure)
  - Test: Feature labeling job triggered after extraction completes
  - **File:** `backend/tests/integration/test_extraction_flow_e2e.py` (NEW)

**Acceptance Criteria:**
- ✅ Integration test coverage increased from 20% to ≥50%
- ✅ WebSocket emission flows tested end-to-end (backend → frontend)
- ✅ Training job lifecycle tested for all status transitions
- ✅ Dataset tokenization flow tested with realistic scenarios
- ✅ Feature extraction flow tested with error handling

---

## Phase 7: E2E Test Foundation (Optional - Backlog)

**Priority:** P2 - Nice to have, not blocking production
**Status:** BACKLOG
**Estimated Time:** 12-16 hours

### Overview
Establish E2E testing infrastructure with Playwright or Cypress. Cover critical user workflows in browser.

### Tasks (Deferred)

- [ ] **7.1 Set up E2E testing framework** (Playwright recommended)
- [ ] **7.2 E2E test: Training workflow** (create → configure → start → monitor → complete)
- [ ] **7.3 E2E test: Dataset workflow** (download → tokenize → browse samples)
- [ ] **7.4 E2E test: Model workflow** (download → extract features → browse features)
- [ ] **7.5 E2E test: System monitoring** (real-time GPU/CPU/memory updates)

**Note:** E2E tests are valuable but not required for initial production deployment. Focus on unit and integration tests first.

---

## Testing Infrastructure

### Required Tools
- **Backend:** pytest, pytest-asyncio, pytest-mock, pytest-cov
- **Frontend:** Vitest, @testing-library/react, @testing-library/user-event
- **Mocking:** unittest.mock (Python), vitest/mocks (TypeScript)
- **Coverage:** pytest --cov (backend), vitest --coverage (frontend)
- **CI/CD:** GitHub Actions or similar for automated test runs

### Test Data Strategy
- **Fixtures:** Reusable test data in `backend/tests/fixtures/` and `frontend/tests/fixtures/`
- **Factories:** Factory functions for generating test objects with realistic data
- **Database:** Use test database with Alembic migrations applied
- **Isolation:** Each test should be independent (no shared state)

### Mocking Strategy
- **External APIs:** Mock HuggingFace, GPT API calls
- **File System:** Mock file operations where appropriate
- **Database:** Use real test database for integration tests, mock for unit tests
- **WebSocket:** Mock Socket.IO server for frontend tests
- **Celery:** Mock task execution for unit tests, use real worker for integration tests

---

## Timeline & Milestones

### Week 1: Foundation (28-36 hours)
- **Day 1-2:** Phase 1 - Fix failing tests (12-16h)
- **Day 3-5:** Phase 2 - WebSocket reliability tests (16-20h)

**Milestone:** All tests passing, WebSocket reliability verified

### Week 2: Feature Coverage (32-44 hours)
- **Day 1-2:** Phase 3 - Multi-tokenization tests (12-16h)
- **Day 3-4:** Phase 4 - Feature labeling tests (8-12h)
- **Day 4-5:** Phase 5 - Unit test expansion start (12-16h of 20-24h total)

**Milestone:** New features tested, unit coverage ≥60%

### Week 3: Completion (24-28 hours)
- **Day 1-2:** Phase 5 - Unit test expansion complete (remaining 8-12h)
- **Day 3-5:** Phase 6 - Integration test expansion (16-20h)

**Milestone:** Unit coverage ≥70%, Integration coverage ≥50%, production ready

---

## Success Metrics

### Coverage Targets
- ✅ Backend unit test coverage: **≥70%** (from 40%)
- ✅ Frontend unit test coverage: **≥70%** (from ~40%)
- ✅ Backend integration test coverage: **≥50%** (from 20%)
- ✅ Overall test pass rate: **100%** (0 failing tests)

### Quality Metrics
- ✅ All critical flows tested (training, tokenization, extraction, labeling)
- ✅ WebSocket reliability verified (connection, reconnection, message delivery)
- ✅ Error recovery flows tested (OOM, network failures, database errors)
- ✅ Test execution time: **<5 minutes** for full suite

### Production Readiness Checklist
- ✅ No failing tests in CI/CD pipeline
- ✅ Coverage thresholds enforced in CI/CD
- ✅ Test documentation complete (how to run, how to add new tests)
- ✅ Mocking patterns documented for future developers

---

## Risk Assessment

### High Risk - Phase 1
**Risk:** Cannot proceed with other phases if failing tests block development
**Mitigation:** Prioritize Phase 1, allocate extra time if needed (up to 20 hours)

### Medium Risk - WebSocket Tests (Phase 2)
**Risk:** WebSocket testing requires specific infrastructure (Socket.IO mock server)
**Mitigation:** Use existing libraries (socket.io-mock), follow documentation examples

### Medium Risk - Time Estimates
**Risk:** Time estimates are rough, actual time may vary by ±30%
**Mitigation:** Buffer built into estimates (84-108 hours = 24-hour range)

### Low Risk - Test Coverage Target
**Risk:** May not reach exactly 70% coverage even with all tasks complete
**Mitigation:** Focus on critical paths first, accept 65-70% as success threshold

---

## Post-Completion Actions

### Documentation
- [ ] Update CLAUDE.md with new test coverage numbers
- [ ] Document testing patterns in `docs/TESTING.md`
- [ ] Create developer guide for writing new tests
- [ ] Update CI/CD configuration with coverage thresholds

### Continuous Improvement
- [ ] Set up coverage tracking over time (trend analysis)
- [ ] Add coverage badge to README
- [ ] Schedule quarterly test review (identify gaps, remove obsolete tests)
- [ ] Monitor test execution time (optimize slow tests)

---

## Related Files
- **Code Review:** `.claude/context/sessions/comprehensive_code_review_2025-11-09.md`
- **Task Plan:** `0xcc/tasks/TASK_LIST_UPDATE_PLAN_2025-11-09.md`
- **Progress Tracking:** `0xcc/tasks/SUPP_TASKS|Progress_Architecture_Improvements.md` (HP-2 completed)
- **Test Files:** `backend/tests/`, `frontend/tests/`

---

**Task List Status:** PLANNED - Ready for Execution ⏳
**Estimated Duration:** 84-108 hours (10-14 developer days)
**Target Completion:** 2-3 weeks with focused effort
