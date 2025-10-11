# GitHub Issues for Deferred Features

**Generated:** 2025-10-11
**Source:** Product Decision Log (PDL-001, PDL-002, PDL-003)
**Related Document:** `0xcc/docs/Product_Decision_Log.md`

This document contains structured issue templates for all deferred features from the Dataset Management MVP. These can be manually created in GitHub or imported via automation.

---

## Issue #1: Full-Text Search & Advanced Filtering for Dataset Samples

**Title:** Add full-text search and advanced filtering to sample browser (FR-4.6, FR-4.7, FR-4.10)

**Labels:** `enhancement`, `phase-12`, `dataset-management`, `deferred-mvp`

**Milestone:** Phase 12 (Post-Core Features)

**Priority:** P2 (Nice-to-Have)

**Estimated Effort:** 15-22 hours

### Description

Add full-text search and advanced filtering capabilities to the dataset sample browser, allowing users to quickly find specific samples based on text content and metadata attributes.

### Background

**Deferred from MVP (PDL-001)** - The MVP shipped with pagination-only browsing (20 samples/page). User research showed 95% of users accomplish their goals by scanning the first 100 samples (5 pages), making search a quality-of-life improvement rather than a functional requirement.

**Time Saved by Deferral:** 15-22 hours, reallocated to code quality improvements (Tasks 13.1-13.8).

### Requirements (from PRD)

**FR-4.6: Full-Text Search**
- Search input field in sample browser header
- Search across `text` field in dataset samples
- Highlight matching terms in results
- Clear search button to reset view

**FR-4.7: Advanced Filtering**
- Filter by metadata fields (e.g., `length`, `label`, custom fields)
- Multiple filter criteria with AND/OR logic
- Visual filter chips showing active filters
- Filter persistence across page navigation

**FR-4.10: PostgreSQL GIN Indexes**
- Create GIN index on `text` column for full-text search
- Create GIN index on `metadata` JSONB column for metadata filtering
- Query optimization for search performance

### Current Workaround

Users browse paginated samples (20/page) and manually scan for relevant examples. For datasets with 1000+ samples, this requires multiple page loads but is adequate for the primary use case of "verify dataset quality."

### Success Criteria for Implementation

Implement this feature **only if**:
- Users report > 10 requests for search functionality
- User feedback indicates pagination is a significant pain point
- Analytics show 20%+ of users browsing > 10 pages per session

### Implementation Guidance

**Option A: PostgreSQL Full-Text Search**
- Pros: No new infrastructure, integrated with existing DB
- Cons: Complex setup, requires GIN indexes and ts_vector columns
- Estimated: 13-16 hours

**Option B: Elasticsearch Integration**
- Pros: Powerful search, faceted filtering, fast
- Cons: New infrastructure, deployment complexity, memory overhead
- Estimated: 18-22 hours

**Recommended:** Start with Option A (PostgreSQL) to validate usage before adding Elasticsearch.

### Technical Design

**Backend Changes:**
```python
# backend/src/api/v1/endpoints/datasets.py
@router.get("/{dataset_id}/samples")
async def get_samples(
    dataset_id: int,
    page: int = 1,
    limit: int = 20,
    search: Optional[str] = None,  # NEW
    filters: Optional[str] = None,  # NEW (JSON-encoded)
    db: Session = Depends(get_db)
):
    # Add full-text search query
    # Add metadata filtering
    pass
```

**Frontend Changes:**
```typescript
// frontend/src/components/datasets/DatasetDetailModal.tsx
function SamplesTab() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({});

  // Add search input and filter UI
  // Update API call to include search/filters
}
```

**Database Migration:**
```sql
-- Create GIN index for full-text search
CREATE INDEX idx_samples_text_gin ON dataset_samples
USING GIN (to_tsvector('english', text));

-- Create GIN index for metadata filtering
CREATE INDEX idx_samples_metadata_gin ON dataset_samples
USING GIN (metadata);
```

### References

- **Product Decision Log:** `0xcc/docs/Product_Decision_Log.md` (PDL-001)
- **Feature PRD:** `0xcc/prds/001_FPRD|Dataset_Management.md` (FR-4.6, FR-4.7, FR-4.10)
- **Technical Design:** `0xcc/tdds/001_FTDD|Dataset_Management.md` (Section 3.3)
- **Current Implementation:** `frontend/src/components/datasets/DatasetDetailModal.tsx:179-289`

### Related Issues

- None (first implementation of search/filtering)

---

## Issue #2: Bulk Dataset Deletion

**Title:** Add bulk deletion for multiple datasets (FR-6.7)

**Labels:** `enhancement`, `future`, `dataset-management`, `deferred-mvp`

**Milestone:** Post-MVP Enhancements

**Priority:** P3 (Low Priority)

**Estimated Effort:** 5-7 hours

### Description

Add ability to select multiple datasets and delete them in a single operation, improving efficiency for bulk cleanup scenarios.

### Background

**Deferred from MVP (PDL-002)** - The MVP shipped with single-dataset deletion only. Deletion is a rare operation (< 1% of all operations), and single deletion handles all current use cases (just slightly slower for edge cases).

**Time Saved by Deferral:** 5-7 hours, reallocated to metadata validation improvements (Task 13.3).

### Requirements (from PRD)

**FR-6.7: Bulk Deletion**
- Checkbox selection for multiple datasets
- "Select All" / "Deselect All" buttons
- Bulk delete button with confirmation dialog
- Confirmation shows count of selected datasets
- Transactional deletion (all-or-nothing or best-effort with error reporting)
- Progress indication for batch operation

### Current Implementation

**Single Deletion:** Users delete datasets one at a time via the delete button in each dataset card. Confirmation dialog prevents accidental deletion. File cleanup and database cascade work correctly.

**Code:** `frontend/src/components/datasets/DatasetCard.tsx:152-180`

### Success Criteria for Implementation

Implement this feature **only if**:
- Users report > 5 requests for bulk deletion
- User feedback indicates single deletion is a pain point
- Scale increases (users managing 50+ datasets)

**Note:** On Jetson edge devices, users typically maintain 5-15 datasets max, making bulk deletion unlikely to be needed.

### Implementation Guidance

**Backend Changes:**
```python
# backend/src/api/v1/endpoints/datasets.py
@router.post("/bulk-delete")
async def bulk_delete_datasets(
    dataset_ids: List[int],
    db: Session = Depends(get_db)
):
    """Delete multiple datasets in a transaction."""
    try:
        for dataset_id in dataset_ids:
            dataset = await dataset_service.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            await dataset_service.delete_dataset(db, dataset_id)
        db.commit()
        return {"deleted_count": len(dataset_ids)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
```

**Frontend Changes:**
```typescript
// frontend/src/components/panels/DatasetsPanel.tsx
function DatasetsPanel() {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  const handleBulkDelete = async () => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/datasets/bulk-delete`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_ids: selectedIds }),
      }
    );
    // Handle response and refresh list
  };

  // Add checkboxes to DatasetCard
  // Add bulk action buttons
}
```

### Design Considerations

**Error Handling:**
- **All-or-nothing:** Rollback entire operation if any deletion fails (simpler, safer)
- **Best-effort:** Delete what we can, report errors for failures (more complex, better UX)

**Recommended:** Start with all-or-nothing for simplicity.

**UX Risk Mitigation:**
- Clear confirmation dialog: "Delete 5 datasets? This action cannot be undone."
- Show list of dataset names in confirmation
- Require explicit confirmation (not just "OK" button)

### References

- **Product Decision Log:** `0xcc/docs/Product_Decision_Log.md` (PDL-002)
- **Feature PRD:** `0xcc/prds/001_FPRD|Dataset_Management.md` (FR-6.7)
- **Current Implementation:** `backend/src/api/v1/endpoints/datasets.py:197-212`

### Related Issues

- None

---

## Issue #3: Local Dataset Upload Support

**Title:** Add support for uploading local datasets (JSONL, CSV, Parquet, text files) (FR-7)

**Labels:** `feature`, `phase-2`, `dataset-management`, `deferred-mvp`, `high-complexity`

**Milestone:** Phase 2 (Future Major Feature)

**Priority:** P2 (Moderate Priority, validated by user need)

**Estimated Effort:** 40-60 hours

### Description

Add comprehensive local dataset upload functionality, allowing users to upload datasets in various formats (JSONL, CSV, Parquet, plain text) without requiring HuggingFace integration.

### Background

**Deferred from MVP (PDL-003)** - The MVP shipped with HuggingFace-only integration, which covers 95% of research use cases (50,000+ public datasets available). This is the largest deferral, saving 40-60 hours of implementation time.

**Time Saved by Deferral:** 40-60 hours, reallocated to core features and quality improvements that delivered a production-ready system (8.5/10 quality score).

**Why This Is Complex:** Local upload requires file validation, format detection, schema inference, data quality checks, character encoding detection, memory-efficient streaming, security hardening, and extensive error handling—4x more complex than HuggingFace integration.

### Requirements (from PRD)

**FR-7: Local Dataset Upload**
- File upload endpoint supporting multipart/form-data
- Format detection: JSONL, CSV, Parquet, plain text
- Schema validation and inference
- Character encoding detection (UTF-8, Latin-1, etc.)
- Data quality checks (empty strings, null values, duplicates)
- Memory-efficient streaming for large files (1GB+)
- File size limits and validation
- Progress tracking for uploads
- Transaction safety with rollback on failure
- Security: malicious file detection, path traversal prevention

### Current Workaround

**Users can upload datasets to HuggingFace as private datasets:**
1. Create private dataset on HuggingFace Hub (10 minutes, free)
2. Upload dataset files via HuggingFace web interface
3. Download via miStudio using access token (built-in support)
4. Benefit: Leverages HuggingFace's robust infrastructure

**Documented in:** User guide should include this workflow clearly.

### Success Criteria for Implementation

Implement this feature **only if**:
- Users report > 20 requests for local upload
- Clear use cases emerge (e.g., "We have 50GB of proprietary chat logs")
- Enterprise interest validates the effort (funding/adoption)

### Alternative Approaches (Evaluate First)

**Option A: HuggingFace Local Directory Support**
- Use `datasets.load_dataset('json', data_dir='...')` for local files
- Simpler than custom upload (15-20 hours)
- Still needs validation but leverages HF library
- **Recommended:** Evaluate this first before building custom upload

**Option B: Custom Upload (Full Implementation)**
- Complete file upload system as described
- Maximum flexibility but highest complexity (40-60 hours)

**Option C: Partner with HuggingFace**
- Leverage HuggingFace Enterprise features
- Offload complexity to battle-tested infrastructure
- May require licensing/partnership agreement

### Technical Design (If Implemented)

**Backend Endpoint:**
```python
# backend/src/api/v1/endpoints/datasets.py
@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # 1. Validate file size (< 10GB for Jetson)
    # 2. Detect format (JSONL/CSV/Parquet/text)
    # 3. Detect encoding (UTF-8/Latin-1/etc.)
    # 4. Stream validation (check schema, data quality)
    # 5. Save to /data/datasets/local/{dataset_id}/
    # 6. Create database record
    # 7. Return dataset object
    pass
```

**Format Detection Logic:**
```python
def detect_format(file: UploadFile) -> str:
    # Check file extension first
    if file.filename.endswith('.jsonl'):
        return 'jsonl'
    elif file.filename.endswith('.csv'):
        return 'csv'
    elif file.filename.endswith('.parquet'):
        return 'parquet'

    # Fall back to content detection
    # Read first 1KB and inspect
    pass
```

**Security Validation:**
```python
def validate_upload_security(file: UploadFile):
    # 1. Check file size (max 10GB)
    # 2. Validate MIME type
    # 3. Scan for path traversal attempts
    # 4. Check for zip bombs (compressed files)
    # 5. Validate file is actually the claimed format
    pass
```

**Frontend Upload Component:**
```typescript
// frontend/src/components/datasets/LocalUploadForm.tsx
function LocalUploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);

    // Use XMLHttpRequest for progress tracking
    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        setUploadProgress((e.loaded / e.total) * 100);
      }
    });
    // Send request...
  };
}
```

### Edge Device Considerations

**Jetson Orin Nano Constraints:**
- 8GB RAM: Must stream files, cannot load entire dataset into memory
- 256GB storage: Need to validate disk space before upload
- Limited bandwidth: Large uploads over USB may be slow

**Recommendation:** For very large datasets (> 10GB), downloading from HuggingFace may actually be faster than local upload.

### Security Checklist

- [ ] File size validation (prevent OOM)
- [ ] MIME type validation (prevent arbitrary file execution)
- [ ] Path traversal protection (sanitize filenames)
- [ ] Zip bomb detection (compressed file limits)
- [ ] Format validation (reject malicious files)
- [ ] Character encoding validation (prevent injection attacks)
- [ ] Disk space validation (prevent disk exhaustion)
- [ ] Rate limiting (prevent abuse)

### References

- **Product Decision Log:** `0xcc/docs/Product_Decision_Log.md` (PDL-003)
- **Feature PRD:** `0xcc/prds/001_FPRD|Dataset_Management.md` (FR-7)
- **Technical Design:** `0xcc/tdds/001_FTDD|Dataset_Management.md` (Section 6.2 - future consideration)
- **HuggingFace Integration Reference:** `backend/src/workers/dataset_tasks.py:29-126`

### Related Issues

- Consider: "Add support for HuggingFace local directory loading" (simpler alternative)

---

## Issue Summary Table

| Issue | Feature | Priority | Effort | Phase | Criteria for Implementation |
|-------|---------|----------|--------|-------|------------------------------|
| #1 | Full-Text Search & Filtering | P2 | 15-22h | Phase 12 | > 10 user requests OR 20%+ users browse > 10 pages |
| #2 | Bulk Deletion | P3 | 5-7h | Future | > 5 user requests OR users manage 50+ datasets |
| #3 | Local Upload | P2 | 40-60h | Phase 2 | > 20 user requests OR enterprise interest |

**Total Deferred Effort:** 60-89 hours

**Value of Deferral:** Time reallocated to core features and quality improvements, resulting in production-ready Dataset Management system (8.5/10 quality score, 23/23 tests passing, 95% requirements coverage).

---

## Next Steps

1. **Copy these issue templates** to GitHub Issues interface manually, OR
2. **Install GitHub CLI** (`gh`) and use automation script to create issues, OR
3. **Wait for user validation** before creating issues (users may not want these tracked in GitHub)

**When to implement deferred features:**
- Monitor user feedback during MVP testing
- Track analytics (page browsing patterns, deletion frequency, upload requests)
- Re-evaluate in Phase 12 review based on actual usage data

---

**Document Version:** 1.0
**Created:** 2025-10-11
**Related Documents:**
- Product Decision Log: `0xcc/docs/Product_Decision_Log.md`
- Feature PRD: `0xcc/prds/001_FPRD|Dataset_Management.md`
- Comprehensive Review: `0xcc/docs/Dataset_Management_Comprehensive_Review.md`
