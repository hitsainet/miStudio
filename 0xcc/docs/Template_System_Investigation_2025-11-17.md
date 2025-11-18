# Template System Investigation - November 17, 2025

## Issue Reported
User reported that their custom "AI_ASSISTED_V1" prompt template was not being used during labeling jobs. Instead, a verbose hardcoded default prompt was being sent to the API.

## Investigation Results

### 1. Template Storage ✅
- Template "AI_ASSISTED_V1" exists in database with ID `lpt_bfed24b682904421`
- Contains correct system message and user prompt template
- System message length: 976 characters
- User prompt template length: 677 characters

### 2. Template Selection ✅
- Frontend correctly passes `prompt_template_id` in labeling requests
- Backend correctly stores `prompt_template_id` in labeling_jobs table
- Most recent labeling job confirms: `prompt_template_id = lpt_bfed24b682904421`

### 3. Template Loading ✅
- Backend code in `labeling_service.py` (lines 590-608) correctly:
  - Checks if `labeling_job.prompt_template_id` exists
  - Queries database for template
  - Extracts `system_message`, `user_prompt_template`, `temperature`, `max_tokens`, `top_p`
  - Passes these to OpenAILabelingService constructor

### 4. Template Usage ✅
- OpenAILabelingService correctly checks if custom template exists (line 307)
- If `self.user_prompt_template` is set, uses `_build_prompt_from_template()`
- If not set, falls back to `_build_prompt()` with hardcoded default

### 5. Actual API Requests ✅
Verified actual saved requests in `/tmp_api/20251117_120632_label_extr_20251116_201719_train_36_20251117_120430/`:
- System message matches AI_ASSISTED_V1 template exactly
- User prompt matches AI_ASSISTED_V1 template exactly
- Tokens table is correctly inserted and formatted
- No hardcoded default prompt is present

## Issue Fixed

### Placeholder Format Inconsistency
- **Problem**: Template stored in database used `{{tokens_table}}` (double braces)
- **Expected**: Code replaces `{tokens_table}` (single braces)
- **Impact**: Minimal - tokens were still being inserted correctly, but format was inconsistent
- **Fix**: Updated template to use `{tokens_table}` (single braces) for consistency

```sql
UPDATE labeling_prompt_templates
SET user_prompt_template = REPLACE(user_prompt_template, '{{tokens_table}}', '{tokens_table}')
WHERE name = 'AI_ASSISTED_V1';
```

### Verification
All templates now use single-brace format consistently:
- ✅ Default Labeling Prompt - single-braces
- ✅ Custom_V1 - single-braces
- ✅ Enhanced_Custom_v1 - single-braces
- ✅ AI_ASSISTED_V1 - single-braces

## Conclusion

**The template system is working correctly.**

The most recent labeling jobs are using the AI_ASSISTED_V1 template as expected. If the user reported seeing a different prompt, they may have been:
1. Looking at an older saved request from before template support was added
2. Looking at a request from a labeling job where no template was selected (uses default)
3. Experiencing a caching issue that has since been resolved

## Recommendation

User should:
1. Clear browser cache if frontend is showing stale data
2. Start a new labeling job with AI_ASSISTED_V1 template selected
3. Verify the saved API requests in `/tmp_api/` show the correct template
4. Check that `{tokens_table}` placeholder is correctly replaced with actual tokens

## Files Verified

- `/home/x-sean/app/miStudio/backend/src/services/labeling_service.py` (lines 590-627, 705-743)
- `/home/x-sean/app/miStudio/backend/src/services/openai_labeling_service.py` (lines 102-103, 307-312, 315-329, 468-525)
- `/home/x-sean/app/miStudio/frontend/src/components/labeling/StartLabelingButton.tsx` (lines 40, 164, 214-229)
- Database table: `labeling_prompt_templates`
- Database table: `labeling_jobs`
- Saved API requests: `/tmp_api/20251117_120632_label_extr_20251116_201719_train_36_20251117_120430/`

## Date
November 17, 2025
