# Product Engineer Agent

**Role:** Deep context development and business alignment
**Focus:** Requirements, user stories, business logic, product vision

## Current Analysis

### Context Completeness
**Last Assessed:** 2025-10-22
**Status:** EXCELLENT - Core features well-implemented

**Requirements Coverage:**
- [x] User stories defined with acceptance criteria
- [x] Business rules documented and understood
- [x] Edge cases identified and planned (retry logic, error handling)
- [x] Integration requirements mapped
- [x] Success metrics defined (progress tracking, real-time updates)

**Context Gaps Identified:**
1. Resource utilization not correlated with active jobs (users can't see which job is using which GPU)
2. No historical progress visualization for performance debugging
3. Missing comparative analysis across training runs

### Product Health Check
- **User Story Coverage:** 90% complete
- **Business Logic Clarity:** CLEAR
- **Integration Readiness:** READY

### Current Recommendations
**High Priority:**
1. Integrate system resource monitoring with job progress displays (show "Training X using GPU 0 - 85%")
2. Add unified operations dashboard showing all active jobs + resource usage

**Medium Priority:**
1. Add progress history visualization for debugging slow training runs
2. Implement training run comparison features for optimization

**Low Priority:**
1. Add resource usage predictions based on job parameters
2. Implement job queuing recommendations based on available resources

### Session Context
**Working On:** Progress tracking & resource monitoring architecture review
**Next Steps:**
1. Design unified operations dashboard UX
2. Plan resource-job correlation UI patterns
3. Gather user feedback on current progress visibility

**Blockers:**
- None - current implementation meets core requirements well

**Notes:**
Recent comprehensive architecture review shows excellent user-facing progress tracking. All job types provide real-time feedback with meaningful context. Main opportunity is better integration between resource monitoring and job execution visibility.


---

## Usage with Claude Code

### Loading This Agent Context
```markdown
@.claude/context/agents/product_engineer.md
@CLAUDE.md
```

### Integration with Commands
- Use with `/analyze product [scope]`
- Update after `/feature` command usage
- Reference in `/review` workflows
- Include in `/collaborate` sessions

### Agent Activation Phrase
"Load the Product Engineer agent context for requirements analysis"

---
*Update this context when working on product/requirements analysis*
