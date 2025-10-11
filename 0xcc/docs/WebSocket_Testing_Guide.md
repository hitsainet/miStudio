# WebSocket Testing Guide

**Feature:** Dataset Management - Real-Time Progress Updates
**WebSocket URL:** `ws://localhost:8001` (or `ws://mistudio.mcslab.io` in production)
**Created:** 2025-10-11
**Status:** Ready for Manual Testing

---

## Overview

This guide walks through manual testing of WebSocket functionality for dataset download and tokenization progress updates.

## Prerequisites

1. **Services Running:**
   ```bash
   # Backend API (port 8000)
   cd /home/x-sean/app/miStudio/backend
   source venv/bin/activate
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

   # Celery Worker (background jobs)
   celery -A src.core.celery_app worker --loglevel=info

   # Frontend (port 3000)
   cd /home/x-sean/app/miStudio/frontend
   npm run dev
   ```

2. **Browser:** Chrome or Firefox with DevTools
3. **Dataset:** Small test dataset (e.g., `roneneldan/TinyStories` with split='train[:1%]')

---

## Test 1: WebSocket Connection Verification (11.7)

### Purpose
Verify that the frontend successfully connects to the WebSocket server and maintains the connection.

### Steps

1. **Open Application**
   - Navigate to `http://localhost:3000`
   - Open Browser DevTools (F12)
   - Go to **Console** tab

2. **Check Connection Messages**
   - Look for Socket.IO connection logs in console:
   ```javascript
   // Expected messages:
   "Socket.IO connected: <socket-id>"
   "WebSocket connected"
   ```

3. **Verify in Network Tab**
   - Switch to **Network** tab
   - Filter by **WS** (WebSocket)
   - You should see:
     - Connection to `ws://localhost:8001` (or configured URL)
     - Status: `101 Switching Protocols`
     - Type: `websocket`

4. **Check Connection Object** (Console)
   ```javascript
   // In Console, the WebSocket manager should be available
   // If using Socket.IO client directly:
   io.sockets  // Should show connected sockets
   ```

### Expected Results
- ✅ Console shows "WebSocket connected"
- ✅ Network tab shows active WebSocket connection
- ✅ No connection errors or repeated reconnection attempts

### Troubleshooting
- **Connection refused**: Verify backend is running on correct port
- **CORS errors**: Check backend CORS configuration allows frontend origin
- **Repeated reconnections**: Backend WebSocket server may be crashing - check backend logs

---

## Test 2: Download Progress Updates (11.8a)

### Purpose
Verify that download progress events are received and UI updates correctly.

### Steps

1. **Start a Download**
   - In the Datasets panel, enter a test dataset:
     - Repository: `roneneldan/TinyStories`
     - Split: `train[:1%]` (small subset for faster testing)
   - Click "Download"

2. **Monitor Console**
   - Watch for WebSocket events in console:
   ```javascript
   // Expected progress events:
   "Joined channel: datasets/<dataset-id>/progress"
   "Progress update received: { progress: 0, status: 'downloading', message: 'Starting download...' }"
   "Progress update received: { progress: 10, status: 'downloading', message: 'Downloading from HuggingFace Hub...' }"
   "Progress update received: { progress: 70, status: 'downloading', message: 'Saving dataset to disk...' }"
   "Progress update received: { progress: 90, status: 'downloading', message: 'Download complete, processing metadata...' }"
   "Tokenization completed: { progress: 100, status: 'ready', message: 'Dataset downloaded successfully' }"
   ```

3. **Monitor UI Elements**
   - Dataset card should show:
     - Status badge: "Downloading" (blue)
     - Progress bar appears
     - Progress bar animates from 0% → 100%
     - Progress percentage updates (e.g., "45.5%")

4. **Check Final State**
   - After completion:
     - Status badge changes to "Ready" (emerald/green)
     - Progress bar disappears
     - Sample count and file size appear

### Expected Results
- ✅ Progress bar shows up immediately when download starts
- ✅ Progress percentage updates smoothly (no jumps backward)
- ✅ Status badge changes: downloading → ready
- ✅ UI shows final metadata (samples, size) after completion

### Troubleshooting
- **No progress updates**: Check Celery worker is emitting events (backend logs)
- **Progress stuck at 0%**: WebSocket channel subscription may have failed
- **Progress jumps**: Multiple workers processing same dataset (check Celery config)

---

## Test 3: Tokenization Progress Updates (11.8b)

### Purpose
Verify tokenization progress tracking with finer granularity (0% → 10% → 20% → 40% → 75% → 95% → 100%).

### Steps

1. **Open Dataset Detail Modal**
   - Click on a ready dataset card
   - Navigate to **Tokenization** tab

2. **Start Tokenization**
   - Select tokenizer: `gpt2`
   - Set max_length: `512`
   - Set stride: `0`
   - Click "Start Tokenization"

3. **Monitor Console**
   - Watch for detailed progress events:
   ```javascript
   // Expected sequence:
   "Joined channel: datasets/<dataset-id>/progress"
   "Progress update received: { progress: 0, message: 'Starting tokenization...' }"
   "Progress update received: { progress: 10, message: 'Loading tokenizer...' }"
   "Progress update received: { progress: 20, message: 'Loading dataset...' }"
   "Progress update received: { progress: 30, message: 'Analyzing dataset schema...' }"
   "Progress update received: { progress: 40, message: 'Tokenizing 50,000 samples...' }"
   // ... batch updates during tokenization (40-75%) ...
   "Progress update received: { progress: 75, message: 'Tokenization complete (50,000 samples)' }"
   "Progress update received: { progress: 80, message: 'Calculating statistics...' }"
   "Progress update received: { progress: 95, message: 'Saving results...' }"
   "Tokenization completed: { progress: 100, message: 'Tokenization complete', statistics: {...} }"
   ```

4. **Monitor UI Elements**
   - Modal should show:
     - Spinning loader icon
     - Progress bar (ProgressBar component)
     - Progress percentage (e.g., "45.2%")
     - Progress message (e.g., "Tokenizing... 25,000/50,000 samples")

5. **Verify Automatic Refresh**
   - After completion, modal should auto-refresh dataset data
   - Statistics tab should now show tokenization stats
   - Tokenization tab should show "Dataset Already Tokenized" banner

### Expected Results
- ✅ Progress updates every few seconds during tokenization
- ✅ Progress messages are descriptive and helpful
- ✅ Progress bar reaches exactly 100% on completion
- ✅ Modal automatically refreshes with new statistics
- ✅ Dataset status changes from "processing" to "ready"

### Troubleshooting
- **Progress stuck at 40%**: Tokenization may have failed - check Celery logs
- **No automatic refresh**: Check `onDatasetUpdate` callback is firing
- **Statistics not showing**: Check metadata was saved to database correctly

---

## Test 4: Reconnection Logic (11.9)

### Purpose
Verify that WebSocket connection automatically reconnects after disconnect and re-subscribes to channels.

### Steps

1. **Establish Baseline**
   - Start a long-running operation (download or tokenization)
   - Verify progress updates are working

2. **Simulate Disconnect - Method 1: Network Tab**
   - Open DevTools → Network tab
   - Right-click the WebSocket connection
   - Select "Block request URL" or "Close"

3. **Simulate Disconnect - Method 2: Kill Backend**
   ```bash
   # In backend terminal, press Ctrl+C to stop server
   # Wait 5 seconds
   # Restart: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Monitor Reconnection**
   - Console should show:
   ```javascript
   "WebSocket disconnected"
   "Attempting to reconnect..."
   "WebSocket connected"
   "Re-subscribing to channels..."
   "Joined channel: datasets/<dataset-id>/progress"
   ```

5. **Verify Progress Resumes**
   - Progress updates should continue after reconnection
   - No progress data should be lost (backend continues processing)

### Expected Results
- ✅ Automatic reconnection within 5 seconds
- ✅ All active channel subscriptions are restored
- ✅ Progress updates resume seamlessly
- ✅ No duplicate progress events
- ✅ User sees notification of disconnection/reconnection (optional)

### Troubleshooting
- **No reconnection**: Check Socket.IO reconnection config in `useWebSocket.ts`
- **Subscriptions not restored**: Check subscription list is persisted across reconnects
- **Duplicate events**: Multiple subscriptions to same channel - check cleanup logic

---

## Test 5: Error Handling (11.10 - Bonus)

### Purpose
Verify that WebSocket error events are handled gracefully.

### Steps

1. **Trigger Tokenization Error**
   - Use invalid tokenizer name: `invalid-tokenizer-12345`
   - Start tokenization

2. **Monitor Error Events**
   ```javascript
   // Expected error event:
   "Tokenization error: { status: 'error', message: 'Failed to load tokenizer...' }"
   ```

3. **Verify UI Error Display**
   - Modal should show error message in red alert box
   - Progress bar should reset or show error state
   - Status badge should change to "Error" (red)

### Expected Results
- ✅ Error events are received via WebSocket
- ✅ UI displays user-friendly error message
- ✅ Dataset status changes to "error"
- ✅ Error persists in dataset card until user takes action

---

## Integration Test Checklist

Use this checklist to verify complete WebSocket functionality:

### Connection
- [ ] WebSocket connects on app load
- [ ] Connection is maintained during idle time
- [ ] Connection survives page navigation (stays connected)
- [ ] Reconnection works after network interruption

### Download Progress
- [ ] Progress events received for download
- [ ] Progress bar updates smoothly (0 → 100%)
- [ ] Status badge updates (downloading → ready)
- [ ] Completion event received
- [ ] Final metadata appears in UI

### Tokenization Progress
- [ ] Progress events received for tokenization
- [ ] Progress messages are descriptive
- [ ] Progress bar shows accurate percentage
- [ ] Statistics tab updates after completion
- [ ] Tokenization tab shows "already tokenized" banner

### Error Handling
- [ ] Error events received via WebSocket
- [ ] UI displays error message
- [ ] Status badge shows "error" state
- [ ] Error message persists in dataset card

### Multiple Operations
- [ ] Multiple concurrent downloads work independently
- [ ] Progress updates don't interfere with each other
- [ ] Each dataset card shows correct progress
- [ ] No "cross-talk" between different datasets

---

## WebSocket Event Reference

### Channel Format
```
datasets/{dataset_id}/progress
```

### Event Types

**progress**
```json
{
  "dataset_id": "uuid",
  "progress": 45.2,
  "status": "downloading" | "processing",
  "message": "Downloading from HuggingFace Hub..."
}
```

**completed**
```json
{
  "dataset_id": "uuid",
  "progress": 100.0,
  "status": "ready",
  "message": "Dataset downloaded successfully",
  "num_samples": 50000,
  "size_bytes": 1073741824,
  "statistics": { ... }
}
```

**error**
```json
{
  "dataset_id": "uuid",
  "status": "error",
  "message": "Download failed: Invalid repository ID"
}
```

---

## Debugging Tips

### Enable Verbose Logging

**Frontend (Console):**
```javascript
localStorage.setItem('debug', 'socket.io-client:*');
// Reload page to see detailed Socket.IO logs
```

**Backend (Celery Worker):**
```bash
celery -A src.core.celery_app worker --loglevel=debug
```

### Check WebSocket Health

**Browser Console:**
```javascript
// Check if Socket.IO is connected
if (window.io) {
  console.log('Socket ID:', window.io.id);
  console.log('Connected:', window.io.connected);
}
```

### Verify Channel Subscriptions

**Backend Logs:**
```python
# Look for:
"WebSocket emit: progress to datasets/{id}/progress - Status: 200"
"Joined channel: datasets/{id}/progress"
```

---

## Common Issues & Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No connection | Backend not running | Start backend server |
| CORS error | Frontend origin not allowed | Add to CORS_ORIGINS in config |
| No progress updates | Not subscribed to channel | Check `subscribe()` calls in useEffect |
| Progress stuck | Celery worker crashed | Check Celery logs, restart worker |
| Multiple connections | Component re-rendering | Add proper cleanup in useEffect |
| Events duplicated | Multiple subscriptions | Use `unsubscribe()` in cleanup |

---

## Success Criteria

WebSocket functionality is considered **fully tested** when:

1. ✅ Connection established on app load
2. ✅ Download progress updates received and displayed
3. ✅ Tokenization progress updates received and displayed
4. ✅ Reconnection works after disconnect
5. ✅ Error events handled gracefully
6. ✅ Multiple concurrent operations work independently
7. ✅ No console errors or warnings
8. ✅ UI updates are smooth and responsive

---

## Next Steps

After completing manual testing:
1. Mark tasks 11.7-11.9 as complete in task list
2. Document any issues found in GitHub issues
3. Proceed to task 11.10: Write integration test for WebSocket flow
4. Consider adding automated E2E tests with Playwright

---

**Tested By:** _________________
**Date:** _________________
**Results:** Pass / Fail / Needs Revision
**Notes:** _________________________________
