# MechInterp Studio (miStudio) - Project Specifications

**Last Updated:** 2025-10-05
**Purpose:** Complete specification package for miStudio development

---

## Quick Start

This directory contains all specifications, architectural documents, and **the reference mock UI implementation** needed to develop miStudio. Start here for any development work.

### Recommended Reading Order

1. **Start Here**: [reference-implementation/Mock-embedded-interp-ui.tsx](./reference-implementation/Mock-embedded-interp-ui.tsx) - **THE UI/UX SPECIFICATION** - The MVP must look and behave exactly like this mock
2. **Core Spec**: [core/miStudio_Specification.md](./core/miStudio_Specification.md) - System overview and technical workflow
3. **Implementation Roadmap**: [implementation/Mock_Frontend-Gap_Closure_Instructions.md](./implementation/Mock_Frontend-Gap_Closure_Instructions.md) - Comprehensive step-by-step implementation plan
4. **Infrastructure Setup**: Review all files in [infrastructure/](./infrastructure/) for backend services
5. **Type Definitions**: [types/training.types.ts](./types/training.types.ts) - TypeScript interfaces and API contracts

---

## Directory Structure

```
project-specs/
├── README.md (this file)
│
├── reference-implementation/
│   ├── Mock-embedded-interp-ui.tsx
│   │   ⚠️  CRITICAL: This is the UI/UX specification
│   │   The production MVP MUST look and behave exactly like this
│   │   ~4000+ lines of fully functional React component
│   │   Includes:
│   │   - Complete UI for all 5 tabs (Datasets, Models, Training, Features, Steering)
│   │   - Training configuration with advanced hyperparameters
│   │   - Real-time progress indicators
│   │   - Checkpoint management
│   │   - Model architecture viewer
│   │   - Activation extraction config
│   │   - Dataset detail modals
│   │   - Feature discovery interface
│   │   - Steering controls
│   │   - Comprehensive API documentation in comments
│   │
│   └── src/styles/mock-ui.css
│       CSS custom properties for dynamic styling
│       Progress bars, charts, token highlights
│
├── core/
│   └── miStudio_Specification.md
│       Complete technical workflow specification describing:
│       - Phase 1: Dataset Management Pipeline
│       - Phase 2: Model Loading & Activation Extraction
│       - Phase 3: Autoencoder Training (SAE)
│       - Phase 4: Feature Discovery & Analysis
│       - Phase 5: Steering & Intervention
│
├── infrastructure/
│   ├── 000_SPEC|REDIS_GUIDANCE_USECASE.md
│   │   Redis implementation guide covering:
│   │   - Job Queue Backend (Celery/BullMQ)
│   │   - WebSocket Pub/Sub for real-time updates
│   │   - Rate Limiting
│   │   - Distributed Coordination
│   │   - Application-Level Caching
│   │
│   ├── 001_SPEC|Folder_File_Details.md
│   │   Complete folder & file structure for production app:
│   │   - /frontend (React TypeScript structure)
│   │   - /backend (Python FastAPI structure)
│   │   - Component organization patterns
│   │   - ~500-1000 file production structure
│   │
│   ├── 003_SPEC|Postgres_Usecase_Details_and_Guidance.md
│   │   PostgreSQL database specification:
│   │   - 10 core tables (datasets, models, trainings, etc.)
│   │   - JSONB support for flexible schemas
│   │   - Full-text search with GIN indexes
│   │   - Partitioning strategies
│   │   - Alembic migration patterns
│   │
│   └── 004_SPEC|Template_System_Guidance.md
│       Template/preset system specification:
│       - Training Templates (hyperparameter configs)
│       - Extraction Templates (feature extraction settings)
│       - Steering Presets (saved steering configurations)
│       - Favoriting and reusability patterns
│
├── implementation/
│   └── Mock_Frontend-Gap_Closure_Instructions.md
│       **PRIMARY IMPLEMENTATION GUIDE** (4200+ lines)
│       - Document Status & Recent Updates
│       - Cross-references to all other specs
│       - 8 Phases with detailed step-by-step instructions
│       - Infrastructure Stack Overview
│       - Data Flow Examples
│       - Code samples and API contracts
│       - Implementation priority matrix
│
└── types/
    └── training.types.ts
        TypeScript interfaces for:
        - Hyperparameters (training config)
        - TrainingMetrics (real-time monitoring)
        - Training (job state management)
        - Checkpoint (checkpoint metadata)
        - TrainingWebSocketMessage (real-time updates)
        - Backend API endpoint documentation
```

---

## Key Documents Explained

### Core Specifications

#### miStudio_Specification.md
- **What**: Technical workflow specification for the entire system
- **When to use**: Understanding system architecture, workflow phases, and core requirements
- **Key sections**:
  - Dataset Management Pipeline (HuggingFace integration, tokenization, storage)
  - Model Loading & Activation Extraction (quantization, forward hooks)
  - Autoencoder Training (SAE, Skip, Transcoder architectures)
  - Feature Discovery & Analysis (activation mapping, interpretation)
  - Steering & Intervention (feature manipulation for model control)

### Infrastructure Specifications

#### Redis Guidance (000_SPEC|REDIS_GUIDANCE_USECASE.md)
- **What**: Complete Redis implementation patterns
- **When to use**:
  - Setting up job queues for async tasks
  - Implementing real-time WebSocket pub/sub
  - Configuring rate limiting
  - Understanding distributed coordination
- **Critical for**: Training progress updates, long-running tasks, real-time UI updates

#### Folder Structure (001_SPEC|Folder_File_Details.md)
- **What**: Production-ready directory structure
- **When to use**:
  - Starting new frontend/backend projects
  - Organizing components
  - Planning file architecture
- **Critical for**: Moving from mock UI to production structure

#### PostgreSQL Database (003_SPEC|Postgres_Usecase_Details_and_Guidance.md)
- **What**: Complete database schema and patterns
- **When to use**:
  - Designing API endpoints
  - Understanding data relationships
  - Writing database queries
  - Planning migrations
- **Critical for**: All CRUD operations, metadata storage, time-series data

#### Template System (004_SPEC|Template_System_Guidance.md)
- **What**: Save/load configuration system
- **When to use**:
  - Implementing template save/load features
  - Understanding preset workflows
  - Designing template APIs
- **Critical for**: User experience (saving work), experimentation workflows

### Implementation Guide

#### Mock_Frontend-Gap_Closure_Instructions.md
- **What**: THE PRIMARY IMPLEMENTATION DOCUMENT
- **When to use**: Always - this is your step-by-step guide
- **Structure**:
  - **Phase 1 (P0)**: Training Infrastructure - Hyperparameters, metrics, controls, checkpoints
  - **Phase 2 (P0)**: Feature Discovery Core - Extraction, browsing, analysis
  - **Phase 3 (P0)**: Steering Tool - The most critical missing component
  - **Phase 4 (P1)**: Dataset Management Enhancements
  - **Phase 5 (P1)**: Model Management Enhancements
  - **Phase 6 (P2)**: Advanced Visualizations (UMAP, t-SNE, heatmaps)
  - **Phase 7 (P2)**: System Settings & Management
  - **Phase 8 (P3)**: Polish (project management, export/import, onboarding)
- **Each step includes**:
  - What to Add
  - Implementation Details (code samples)
  - Backend API Required
  - Cross-references to other specs

### Reference Implementation

#### Mock-embedded-interp-ui.tsx
- **What**: THE DEFINITIVE UI/UX SPECIFICATION
- **When to use**: ALWAYS - This defines exactly how the MVP should look and behave
- **Critical points**:
  - Every component in this file should be replicated in production
  - All styling, layouts, interactions, and animations should match exactly
  - The file contains ~4000+ lines with comprehensive API documentation in comments
  - This is not a "rough sketch" - it's a pixel-perfect specification
  - Backend APIs should match the contracts documented in the comments
- **Contains**:
  - All 5 main tabs (Datasets, Models, Training, Features, Steering)
  - Complete training workflow with hyperparameters
  - Real-time progress tracking patterns
  - Checkpoint management UI
  - Model architecture visualization
  - Activation extraction configuration
  - Dataset management with HuggingFace integration
  - Feature discovery and analysis interface
  - Steering controls
  - Template/preset system examples

### Type Definitions

#### training.types.ts
- **What**: TypeScript interfaces and API contract documentation
- **When to use**:
  - Writing frontend TypeScript code
  - Validating API responses
  - Understanding data structures
- **Contains**: Full type definitions with validation constraints and backend API documentation

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)            │
│  ┌───────────┬──────────┬───────────┬──────────┬─────────┐ │
│  │ Datasets  │ Models   │ Training  │ Features │Steering │ │
│  └─────┬─────┴────┬─────┴─────┬─────┴────┬─────┴────┬────┘ │
│        │          │           │          │          │       │
│        └──────────┴───────────┴──────────┴──────────┘       │
│                        │                                     │
│                   REST API + WebSocket                       │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────┐
│               Backend (Python FastAPI)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  API Routes                           │  │
│  │  /api/datasets  /api/models  /api/training  /api/... │  │
│  └────┬─────────────────────┬─────────────────────┬─────┘  │
│       │                     │                     │         │
│  ┌────▼────┐         ┌──────▼──────┐       ┌─────▼─────┐  │
│  │         │         │             │       │           │  │
│  │ PostSQL │◄────────┤   Redis     │──────►│  Celery   │  │
│  │         │         │             │       │  Workers  │  │
│  └─────────┘         └─────────────┘       └───────────┘  │
│      │                     │                      │         │
│  Metadata           Job Queue &             GPU Tasks      │
│  JSONB Data         Pub/Sub for            (Training,      │
│  Relations          WebSocket              Feature         │
│                                            Extraction)      │
└──────────────────────────────────────────────────────────────┘
```

**Key Data Flows**:
1. **Frontend → API → Database**: User actions create/read/update database records
2. **Frontend → API → Celery → Redis**: Long tasks queued for async processing
3. **Celery Worker → Redis Pub/Sub → WebSocket → Frontend**: Real-time updates
4. **Database → API → Frontend**: Data retrieval for display

---

## Common Development Workflows

### Starting a New Feature

1. **Check Implementation Guide**: Find your feature in [implementation/Mock_Frontend-Gap_Closure_Instructions.md](./implementation/Mock_Frontend-Gap_Closure_Instructions.md)
2. **Review Architecture**: Check [core/miStudio_Specification.md](./core/miStudio_Specification.md) for workflow context
3. **Check Types**: Look at [types/training.types.ts](./types/training.types.ts) for data structures
4. **Review Infrastructure**: Check relevant files in [infrastructure/](./infrastructure/):
   - Need job queue? → Redis guide
   - Need database table? → PostgreSQL guide
   - Need templates? → Template System guide
5. **Implement**: Follow step-by-step instructions with code samples
6. **Cross-reference**: Use links in Implementation Guide to jump to related specs

### Setting Up Backend Infrastructure

1. **Read**: [infrastructure/001_SPEC|Folder_File_Details.md](./infrastructure/001_SPEC|Folder_File_Details.md) for structure
2. **Database**: Set up PostgreSQL using [infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md)
3. **Redis**: Configure using [infrastructure/000_SPEC|REDIS_GUIDANCE_USECASE.md](./infrastructure/000_SPEC|REDIS_GUIDANCE_USECASE.md)
4. **Templates**: Implement using [infrastructure/004_SPEC|Template_System_Guidance.md](./infrastructure/004_SPEC|Template_System_Guidance.md)

### Understanding Data Flow

1. **Read**: "Data Flow Examples" section in [implementation/Mock_Frontend-Gap_Closure_Instructions.md](./implementation/Mock_Frontend-Gap_Closure_Instructions.md)
2. **Trace**: Follow specific examples:
   - Starting a Training Job (9 steps)
   - Browsing Features (9 steps)
   - Applying Steering (7 steps)

---

## Implementation Priority

### Phase 0: Foundation (Do First)
- Set up PostgreSQL schema (use 003_SPEC)
- Configure Redis (use 000_SPEC)
- Create basic FastAPI structure (use 001_SPEC)

### Phase 1: MVP Core (P0)
1. **Training Infrastructure** (Steps 1.1-1.5 in Implementation Guide)
   - Hyperparameters panel
   - Real-time metrics via WebSocket
   - Checkpoint management

2. **Feature Discovery** (Steps 2.1-2.6 in Implementation Guide)
   - Feature extraction interface
   - Feature browser
   - Max activating examples

3. **Steering Tool** (Steps 3.1-3.6 in Implementation Guide)
   - Steering tab (most critical missing component)
   - Feature selection
   - Comparative output display

### Phase 2: Enhanced UX (P1)
- Dataset management enhancements (Step 4.x)
- Model management enhancements (Step 5.x)

### Phase 3: Advanced Features (P2)
- Visualizations (UMAP, t-SNE, heatmaps) (Step 6.x)
- System settings (Step 7.x)

### Phase 4: Polish (P3)
- Project management, export/import, onboarding (Step 8.x)

---

## Document Cross-References

All documents are heavily cross-referenced. Look for:

- **Specification Reference**: Links to core specification phases
- **Infrastructure Requirements**: Links to infrastructure specs (Redis, PostgreSQL, etc.)
- **Type Definitions**: Links to TypeScript interfaces
- **Gap Analysis**: Identifies what's missing in current implementation

Follow these links to understand full context of any feature.

---

## Notes for Claude Code

### Context Loading
When working on a feature:
1. Load the relevant section from **Mock_Frontend-Gap_Closure_Instructions.md** (primary)
2. Load referenced infrastructure specs (Redis, PostgreSQL, etc.)
3. Load type definitions from **training.types.ts**
4. Reference core specification for workflow understanding

### Code Generation
- Use TypeScript interfaces from `types/training.types.ts`
- Follow component patterns in `001_SPEC|Folder_File_Details.md`
- Match API contracts documented in Implementation Guide
- Use infrastructure patterns from respective SPEC files

### Testing Against Specs
- Verify implementations match "What to Add" sections
- Ensure API calls match "Backend API Required" sections
- Check data structures match TypeScript types
- Validate infrastructure usage matches SPEC guides

---

## Getting Help

- **Unclear workflow?** → Check [core/miStudio_Specification.md](./core/miStudio_Specification.md)
- **Missing implementation details?** → Check [implementation/Mock_Frontend-Gap_Closure_Instructions.md](./implementation/Mock_Frontend-Gap_Closure_Instructions.md)
- **Infrastructure questions?** → Check relevant file in [infrastructure/](./infrastructure/)
- **Type confusion?** → Check [types/training.types.ts](./types/training.types.ts)

All documents are designed to be self-contained but heavily cross-referenced for easy navigation.

---

**Current Status**: Foundation complete (TypeScript warnings fixed, HuggingFace token support added), ready for Phase 1 implementation
