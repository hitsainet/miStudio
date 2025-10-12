# Project: MechInterp Studio (miStudio)

## Current Status
- **Phase:** Implementation - Phase 6: Frontend State Management & UI Components
- **Last Session:** 2025-10-08 - Completed Phase 3 migrations, all backend tests passing
- **Next Steps:** Implement Zustand stores, API client, and Dataset Management Panel UI
- **Active Document:** 0xcc/tasks/001_FTASKS|Dataset_Management.md (Phase 6 ready)
- **Current Feature:** Dataset Management (Core Feature #1 - P0)
- **Test Status:** 23/23 backend tests passing, database migrations complete, API server running
- **Services Status:** Backend (port 8000) ✅, Frontend (port 3000) ✅, PostgreSQL ✅, Redis ✅

## PRIMARY UI/UX REFERENCE

Key aspects (load full file only when needed):
- UI/UX design patterns and visual style
- Component layouts and interactions
- User workflows and navigation
- API contracts and data structures
- Feature completeness and behavior

All implementation MUST match the Mock UI specification exactly.

## Application Startup

### Complete Startup (All Services)
```bash
# ONE COMMAND to start everything:
./start-mistudio.sh

# This starts (in order):
# 1. Docker services (PostgreSQL, Redis, Nginx)
# 2. Celery worker
# 3. Backend (FastAPI on port 8000)
# 4. Frontend (Vite on port 3000)
#
# Access at: http://mistudio.mcslab.io
```

**IMPORTANT**: Before first run, add domain to /etc/hosts:
```bash
sudo bash -c 'echo "127.0.0.1  mistudio.mcslab.io" >> /etc/hosts'
```

### Stop All Services
```bash
./stop-mistudio.sh
```

### Service Status Check
```bash
# Check all services:
docker ps  # Should show: mistudio-postgres, mistudio-redis, mistudio-nginx
lsof -i :8000  # Backend should be running
lsof -i :3000  # Frontend should be running
pgrep -f celery  # Celery worker should be running

# Access points:
# - Main app: http://mistudio.mcslab.io
# - Frontend direct: http://localhost:3000
# - Backend direct: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

## Quick Resume Commands

### Lean Session Start (Recommended)
```bash
# Minimal context loading - most efficient approach
"Please help me resume where I left off"
# This automatically loads: CLAUDE.md + session_state.json

# Load specific current work area only when needed:
# 0xcc/tasks/[current-task-file].md  # The specific task being worked on
```

### On-Demand Loading Strategy
⚠️ **IMPORTANT**: The following files are LARGE (40k+ chars) and should ONLY be loaded when you encounter specific questions. **DO NOT load them automatically at session start.**

```bash
# Load when UI/styling question arises (207k chars):
# 0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx

# Load when business context/feature priority question arises (54k chars):
# 0xcc/prds/000_PPRD|miStudio.md

# Load when architectural decision question arises (72k chars):
# 0xcc/adrs/000_PADR|miStudio.md

# Load when design clarification needed:
# 0xcc/tdds/[feature]_FTDD.md

# Load when implementation guidance needed:
# 0xcc/tids/[feature]_FTID.md
```

### Research Integration
```bash
# Use MCP ref server for contextual research (when available)
/mcp ref search "[context-specific query]"
```

## Housekeeping Commands
```bash
"Please create a checkpoint"        # Save complete state
"Please help me resume"            # Restore context for new session
"My context is getting too large"  # Clean context, restore essentials
"Please save the session transcript" # Save session transcript
"Please show me project status"    # Display current state
```

## Project Standards

### Technology Stack

**Backend:**
- Python 3.10+, FastAPI, PostgreSQL 14+, Redis 7+, Celery
- PyTorch 2.0+, HuggingFace (transformers, datasets), bitsandbytes
- TensorRT for Jetson optimization

**Frontend:**
- React 18+ with TypeScript, Vite, Zustand
- Tailwind CSS (slate dark theme per Mock UI)
- Lucide React icons, D3.js + Recharts
- Socket.IO for real-time updates

**Infrastructure:**
- Docker Compose for development (nginx, postgres, redis, backend, frontend, celery)
- Nginx reverse proxy (port 80, future HTTPS on 443)
- Base URL: http://mistudio.mcslab.io
- systemd for production (Jetson)
- Local filesystem storage (/data/)

### Coding Standards

**Python:**
- Formatter: Black (line length 100)
- Linter: Ruff
- Type Checker: MyPy (strict)
- Docstrings: Google style

**TypeScript:**
- Formatter: Prettier
- Linter: ESLint (Airbnb)
- All components strictly typed

### Naming Conventions

**Python:** `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
**TypeScript:** `camelCase` functions, `PascalCase` components/types, `UPPER_SNAKE_CASE` constants

### Testing

**Backend:** pytest (>80% coverage target)
**Frontend:** Vitest + React Testing Library
**E2E:** Playwright for critical paths

### Git Workflow

**Branches:** `main` (production), `develop` (integration), `feature/*`, `bugfix/*`
**Commits:** Conventional commits (`feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`)
**Review:** All code reviewed, tests pass, no coverage decrease

### File Organization

**Backend:** `app/api/`, `app/services/`, `app/ml/`, `app/db/`, `app/workers/`
**Frontend:** `src/components/`, `src/stores/`, `src/services/`, `src/hooks/`, `src/types/`

### Error Handling

- Backend: FastAPI HTTPException with proper status codes
- Frontend: Try-catch with axios error handling
- Structured error responses with `error.code`, `error.message`, `error.details`

### API Design

- RESTful conventions (GET, POST, PUT, PATCH, DELETE)
- Response format: `{ data, meta }` or `{ error }`
- Status codes: 200, 201, 202, 400, 404, 409, 429, 500, 503
- Pagination: `?page=1&limit=50`
- WebSocket: Socket.IO with rooms per training job

### UI/UX Standards

#### **PRIMARY REFERENCE:** `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`

- Dark theme: slate color palette (bg-slate-950, 900, 800)
- Emerald accents: buttons, success states
- Tailwind utility classes matching Mock UI exactly
- Functional components with TypeScript
- Zustand for global state, local state for UI

### Database Schema

- PostgreSQL with JSONB for flexible metadata
- Time-series metrics in dedicated tables with indexes
- Partitioned tables for large data (feature_activations)
- Foreign keys with CASCADE for data integrity

### Edge Optimization

- Mixed precision training (FP16)
- Gradient accumulation for large effective batches
- Memory-mapped files for datasets/activations
- TensorRT optimization for Jetson inference
- INT8/INT4 quantization via bitsandbytes

### Deployment

**Development:** Docker Compose (nginx, postgres, redis, backend, frontend, celery)
**Production:** systemd service on Jetson with Docker Compose + nginx reverse proxy
**Base URL:** http://mistudio.mcslab.io (port 80)
**Future HTTPS:** Port 443 with SSL certificate
**Alternative:** Native installation (Nginx + PostgreSQL + Redis + Python + Node.js)

## AI Dev Tasks Framework Workflow

### Document Creation Sequence
1. **Project Foundation**
   - `000_PPRD|[project-name].md` → `0xcc/prds/` (Project PRD)
   - `000_PADR|[project-name].md` → `0xcc/adrs/` (Architecture Decision Record)
   - Update this CLAUDE.md with Project Standards from ADR

2. **Feature Development** (repeat for each feature)
   - `[###]_FPRD|[feature-name].md` → `0xcc/prds/` (Feature PRD)
   - `[###]_FTDD|[feature-name].md` → `0xcc/tdds/` (Technical Design Doc)
   - `[###]_FTID|[feature-name].md` → `0xcc/tids/` (Technical Implementation Doc)
   - `[###]_FTASKS|[feature-name].md` → `0xcc/tasks/` (Task List)

### Instruction Documents Reference
- `0xcc/instruct/001_create-project-prd.md` - Creates project vision and feature breakdown
- `0xcc/instruct/002_create-adr.md` - Establishes tech stack and standards
- `0xcc/instruct/003_create-feature-prd.md` - Details individual feature requirements
- `0xcc/instruct/004_create-tdd.md` - Creates technical architecture and design
- `0xcc/instruct/005_create-tid.md` - Provides implementation guidance and coding hints
- `0xcc/instruct/006_generate-tasks.md` - Generates actionable development tasks
- `0xcc/instruct/007_process-task-list.md` - Guides task execution and progress tracking
- `0xcc/instruct/008_housekeeping.md` - Session management and context preservation

## Document Inventory

### Project Level Documents
- ✅ 0xcc/prds/000_PPRD|miStudio.md (Project PRD - Completed 2025-10-05)
- ✅ 0xcc/adrs/000_PADR|miStudio.md (Architecture Decision Record - Completed 2025-10-05)

### Feature Documents
*[Add as features are identified and developed]*

**Example format:**
- ❌ 0xcc/prds/001_FPRD|Feature_A.md (Feature PRD)
- ❌ 0xcc/tdds/001_FTDD|Feature_A.md (Technical Design Doc)
- ❌ 0xcc/tids/001_FTID|Feature_A.md (Technical Implementation Doc)
- ❌ 0xcc/tasks/001_FTASKS|Feature_A.md (Task List)

### Status Indicators
- ✅ **Complete:** Document finished and reviewed
- ⏳ **In Progress:** Currently being worked on
- ❌ **Pending:** Not yet started
- 🔄 **Needs Update:** Requires revision based on changes

## Housekeeping Status
- **Last Checkpoint:** [Date/Time] - [Brief description]
- **Last Transcript Save:** [Date/Time] - [File location in 0xcc/transcripts/]
- **Context Health:** Good/Moderate/Needs Cleanup
- **Session Count:** [Number] sessions since project start
- **Total Development Time:** [Estimated hours]

## Task Execution Standards

### Completion Protocol
- ✅ One sub-task at a time, ask permission before next
- ✅ Mark sub-tasks complete immediately: `[ ]` → `[x]`
- ✅ When parent task complete: Run tests → Stage → Clean → Commit → Mark parent complete
- ✅ Never commit without passing tests
- ✅ Always clean up temporary files before commit

### Commit Message Format
```bash
git commit -m "feat: [brief description]" -m "- [key change 1]" -m "- [key change 2]" -m "Related to [Task#] in [PRD]"
```

### Test Commands
*[Will be defined in ADR, examples:]*
- **Frontend:** `npm test` or `npm run test:unit`
- **Backend:** `pytest` or `python -m pytest` 
- **Full Suite:** `[project-specific command]`

## Code Quality Checklist

### Before Any Commit
- [ ] All tests passing
- [ ] No console.log/print debugging statements
- [ ] No commented-out code blocks
- [ ] No temporary files (*.tmp, .cache, etc.)
- [ ] Code follows project naming conventions
- [ ] Functions/methods have docstrings if required
- [ ] Error handling implemented per ADR standards

### File Organization Rules
*[Will be defined in ADR, examples:]*
- Place test files alongside source files: `Component.tsx` + `Component.test.tsx`
- Follow directory structure from ADR
- Use naming conventions: `[Feature][Type].extension`
- Import statements organized: external → internal → relative
- Framework files in `0xcc/` directory, project files in standard locations

## Context Management

### Session End Protocol
```bash
# 1. Update CLAUDE.md status section
# 2. Create session summary
"Please create a checkpoint"
# 3. Commit progress
git add .
git commit -m "docs: completed [task] - Next: [specific action]"
```

### Context Recovery (If Lost)
```bash
# Mild context loss - files to reference if needed:
# CLAUDE.md
# 0xcc/session_state.json
ls -la 0xcc/*/
# 0xcc/instruct/[current-phase].md

# Severe context loss - files to reference if needed:
# CLAUDE.md
# 0xcc/prds/000_PPRD|[project-name].md
# 0xcc/adrs/000_PADR|[project-name].md
ls -la 0xcc/*/
# 0xcc/instruct/
```

### Resume Commands for Next Session
```bash
# Standard resume sequence
"Please help me resume where I left off"
# Files are automatically loaded from context - no need to manually load
# Specific next action: [detailed action]
```

## Progress Tracking

### Task List Maintenance
- Update task list file after each sub-task completion
- Add newly discovered tasks as they emerge
- Update "Relevant Files" section with any new files created/modified
- Include one-line description for each file's purpose
- Distinguish between framework files (0xcc/) and project files (src/, tests/, etc.)

### Status Indicators for Tasks
- `[ ]` = Not started
- `[x]` = Completed
- `[~]` = In progress (use sparingly, only for current sub-task)
- `[?]` = Blocked/needs clarification

### Session Documentation
After each development session, update:
- Current task position in this CLAUDE.md
- Any blockers or questions encountered
- Next session starting point
- Files modified in this session (both 0xcc/ and project files)

## Implementation Patterns

### Error Handling
*[Will be defined in ADR - placeholder for standards]*
- Use project-standard error handling patterns from ADR
- Always handle both success and failure cases
- Log errors with appropriate level (error/warn/info)
- User-facing error messages should be friendly

### Testing Patterns
*[Will be defined in ADR - placeholder for standards]*
- Each function/component gets a test file
- Test naming: `describe('[ComponentName]', () => { it('should [behavior]', () => {})})`
- Mock external dependencies
- Test both happy path and error cases
- Aim for [X]% coverage per ADR standards

## Debugging Protocols

### When Tests Fail
1. Read error message carefully
2. Check recent changes for obvious issues
3. Run individual test to isolate problem
4. Use debugger/console to trace execution
5. Check dependencies and imports
6. Ask for help if stuck > 30 minutes

### When Task is Unclear
1. Review original PRD requirements
2. Check TDD for design intent
3. Look at TID for implementation hints
4. Ask clarifying questions before proceeding
5. Update task description for future clarity

## Feature Priority Order
*From Project PRD - Core Features (P0):*

**MVP Features (Must Have):**
1. Dataset Management Panel (P0) - HuggingFace integration, local ingestion
2. Model Management Panel (P0) - Model downloads, quantization, architecture viewer
3. SAE Training System (P0) - Sparse autoencoder training with real-time monitoring
4. Feature Discovery & Browser (P0) - Extract and analyze features from trained SAEs
5. Model Steering Interface (P0) - Feature-based interventions and comparative generation

**Secondary Features (P1):**
6. Training Templates & Presets - Save/load training configurations
7. Extraction Templates - Preset activation extraction configs
8. Steering Presets - Save/load steering configurations
9. Advanced Visualizations - UMAP, correlation heatmaps
10. Feature Analysis Tools - Logit lens, ablation studies
11. Checkpoint Auto-Save - Automatic training checkpoints
12. Dataset Statistics Dashboard - Detailed dataset metrics

**Future Features (P3):**
13. Multi-Model Comparison
14. Export & Reporting
15. Collaborative Features
16. Advanced Circuit Analysis

## Session History Log

### Session 1: 2025-10-05 - Project Foundation
- **Accomplished:**
  - Created 0xcc framework directory structure (prds, adrs, tdds, tids, tasks, docs, transcripts, checkpoints, scripts)
  - Created comprehensive Project PRD (000_PPRD|miStudio.md) based on Mock UI specification
  - Updated CLAUDE.md with project name, status, and UI reference priority
  - Established Mock UI as PRIMARY reference for all implementation
- **Next:** Create Architecture Decision Record using 0xcc/instruct/002_create-adr.md
- **Files Created:**
  - 0xcc/prds/000_PPRD|miStudio.md (14,000+ lines)
  - Updated CLAUDE.md with project context
- **Duration:** ~2 hours
- **Key Decision:** Mock-embedded-interp-ui.tsx is the authoritative UI/UX specification

*[Add new sessions as they occur]*

## Research Integration

### MCP Research Support
When available, the framework supports research integration via:
```bash
# Use MCP ref server for contextual research
/mcp ref search "[context-specific query]"

# Research is integrated into all instruction documents as option B
# Example: "🔍 Research first: Use /mcp ref search 'MVP development timeline'"
```

### Research History Tracking
- Research queries and findings captured in session transcripts
- Key research decisions documented in session state
- Research context preserved across sessions for consistency

## Quick Reference

### 0xcc Folder Structure
```
project-root/
├── CLAUDE.md                       # This file (project memory)
├── 0xcc/                           # XCC Framework directory
│   ├── adrs/                       # Architecture Decision Records
│   ├── docs/                       # Additional documentation
│   ├── instruct/                   # Framework instruction files
│   ├── prds/                       # Product Requirements Documents
│   ├── tasks/                      # Task Lists
│   ├── tdds/                       # Technical Design Documents
│   ├── tids/                       # Technical Implementation Documents
│   ├── transcripts/                # Session transcripts
│   ├── checkpoints/                # Automated state backups
│   ├── scripts/                    # Optional automation scripts
│   ├── session_state.json          # Current session tracking
│   └── research_context.json       # Research history and context
├── src/                            # Your project code
├── tests/                          # Your project tests
└── README.md                       # Project README
```

### File Naming Convention
- **Project Level:** `000_PPRD|ProjectName.md`, `000_PADR|ProjectName.md`
- **Feature Level:** `001_FPRD|FeatureName.md`, `001_FTDD|FeatureName.md`, etc.
- **Sequential:** Use 001, 002, 003... for features in priority order
- **Framework Files:** All in `0xcc/` directory for clear organization
- **Project Files:** Standard locations (src/, tests/, package.json, etc.)

### Emergency Contacts & Resources
- **Framework Documentation:** 0xcc/instruct/000_README.md
- **Current Project PRD:** 0xcc/prds/000_PPRD|miStudio.md
- **PRIMARY UI REFERENCE:** 0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx
- **Tech Specification:** 0xcc/project-specs/core/miStudio_Specification.md
- **Tech Standards:** 0xcc/adrs/000_PADR|miStudio.md
- **Housekeeping Guide:** 0xcc/instruct/008_housekeeping.md

---

**Framework Version:** 1.1
**Last Updated:** 2025-10-05
**Project Started:** 2025-10-05
**Project:** MechInterp Studio (miStudio)
**Structure:** 0xcc framework with MCP research integration