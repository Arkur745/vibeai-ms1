# FastAPI + Celery Task State Bug Fix Guide

## Bug Summary

Your task state polling was returning "failed" or errors instead of proper state transitions because of **three critical issues**:

### 1. **Duplicate & Broken `get_result()` Endpoint** (CRITICAL)

**Problem:**

- `main.py` had TWO `get_result()` functions defined
- Second function OVERRODE the first one
- Second version used undefined `AsyncResult` import → `NameError`

**Impact:** Any result polling would instantly crash

**Fix:**

- Removed duplicate endpoint
- Added explicit import: `from celery.result import AsyncResult`
- Used correct reference: `AsyncResult(task_id, app=celery)`

### 2. **Incomplete Celery Configuration** (CRITICAL)

**Problem:**

- Missing `task_track_started=True` → STARTED state never recorded
- Missing `include=['app.workers.tasks']` → tasks not auto-discovered
- Missing serialization config → result backend couldn't persist state
- Missing `result_expires` → results lost after timeout

**Impact:**

- Task state stuck at PENDING
- State transitions not tracked in Redis
- Task results not stored properly

**Fix in `celery_app.py`:**

```python
# Added full configuration block
celery.conf.update(
    task_track_started=True,           # Track PENDING→STARTED→SUCCESS
    result_expires=3600,               # Keep results for 1 hour
    task_serializer='json',            # Ensure JSON serialization
    accept_content=['json'],
    result_serializer='json',
    # ... other settings
)

celery.conf.include = ['app.workers.tasks']  # Auto-discover tasks
```

### 3. **Exception Handling Swallowing State Updates** (HIGH)

**Problem:**

- `tasks.py` caught exceptions and returned `{"status": "failed", ...}` dict
- This doesn't mark task as FAILURE in Celery - it's still SUCCESS
- Result endpoint sees dict with "status" key, assumes success

**Impact:** Failed tasks reported as succeeded with error dict in result

**Fix:**

- Changed exception handling to `raise` instead of returning error dict
- Added `autoretry_for=(Exception,)` and `retry_kwargs={'max_retries': 2}`
- Celery now properly marks failed tasks with `state="FAILURE"`

---

## What Was Fixed

### File 1: `app/workers/celery_app.py`

**Changes:**

1. Added comprehensive configuration dict with:
   - Task tracking settings
   - Result backend settings
   - Serialization settings (JSON)
   - Queue configuration
   - Retry policy

2. Added `include=['app.workers.tasks']` for auto-discovery

3. Added test `debug_task` for connectivity verification

### File 2: `app/main.py`

**Changes:**

1. Removed duplicate `get_result()` endpoint (line ~320+)
2. Added proper import: `from celery.result import AsyncResult`
3. Enhanced `get_result()` with:
   - Support for all task states: PENDING, STARTED, SUCCESS, FAILURE, RETRY
   - "running" status for STARTED state
   - Detailed error handling with try-catch
   - Comprehensive logging at each state transition
   - Type validation of results
   - HTTPException for error responses

4. Added `/health` endpoint to verify:
   - FastAPI running
   - Celery workers connected
   - Redis accessible

### File 3: `app/workers/tasks.py`

**Changes:**

1. Added `bind=True` to task decorator
2. Added `autoretry_for=(Exception,)` for automatic retries
3. Changed exception handling to `raise` instead of returning error dict
4. Added comprehensive logging with `logger.info()` and `logger.error()`
5. Added result validation before return
6. Added task naming: `name='app.workers.tasks.process_audio_task'`

### File 4: `app/schemas.py`

**Changes:**

1. Made `explanations` optional in `AnalysisResult`
2. Added optional `error` field to `TaskStatusResponse`

---

## Testing the Fix

### 1. **Verify Celery Health**

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "celery": "connected",
  "active_workers": ["celery@hostname"],
  "redis_configured": true
}
```

### 2. **Submit a Task**

```bash
curl -X POST -F "file=@audio.mp3" http://localhost:8000/api/v1/analyze
```

Response:

```json
{
  "task_id": "abc123def456",
  "status": "submitted"
}
```

### 3. **Poll Task Status (Immediately)**

```bash
curl http://localhost:8000/api/v1/result/abc123def456
```

Expected: `"status": "pending"` (while task queued)

### 4. **Poll Task Status (While Running)**

Wait a few seconds, then:

```bash
curl http://localhost:8000/api/v1/result/abc123def456
```

Expected: `"status": "running"` (STARTED state)

### 5. **Poll Task Status (After Completion)**

Wait for task to complete, then:

```bash
curl http://localhost:8000/api/v1/result/abc123def456
```

Expected:

```json
{
  "task_id": "abc123def456",
  "status": "completed",
  "result": {
    "genre": "blues",
    "tempo": 92.29,
    "valence": 4.317,
    "arousal": 4.353,
    "mood": "Melancholic",
    "vibe": "Sad, emotional, reflective"
  }
}
```

### 6. **Check Worker Logs**

Worker should show:

```
TASK STARTED: abc123def456
S3 Key: uploads/uuid.mp3
Downloaded: /path/to/temp_uploads/uuid.mp3
Running inference on audio: /path/to/temp_uploads/uuid.mp3
Prediction completed successfully: {...}
Task abc123def456 completed successfully
Deleted temporary file: /path/to/temp_uploads/uuid.mp3
```

---

## How It Works Now

### State Transitions (Correct)

```
Task Submitted via API
          ↓
    PENDING (task queued, waiting for worker)
          ↓
    STARTED (worker picked up task, running inference)
          ↓
    SUCCESS (inference complete, result returned)
         OR
    FAILURE (exception raised, error recorded)
```

### Redis Storage (Now Working)

When you submit a task:

1. Celery stores task metadata in Redis at key: `celery-task-meta-{task_id}`
2. Initial state: `"state": "PENDING"`
3. When worker starts: `"state": "STARTED"`
4. When complete: `"state": "SUCCESS"` + result stored
5. If error: `"state": "FAILURE"` + exception stored

Your `AsyncResult(task_id, app=celery)` now reads these states correctly.

---

## Startup Commands

### Terminal 1: Start Redis (if not running)

```bash
redis-server
```

### Terminal 2: Start Celery Worker

```bash
cd C:\dev\SAAS\vibeai-ms1
conda activate aimlenv
celery -A app.workers.tasks worker --loglevel=info --pool=solo
```

**Key flags:**

- `-A app.workers.tasks`: App module (ensures tasks.py is imported)
- `--loglevel=info`: Show INFO level logs (includes state transitions)
- `--pool=solo`: Single-process pool (best for Windows + development)

### Terminal 3: Start FastAPI

```bash
cd C:\dev\SAAS\vibeai-ms1
conda activate aimlenv
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Debugging Tips

### If tasks stay "pending" forever:

1. Check worker is running: `curl http://localhost:8000/health` should show active_workers
2. Check Redis: `redis-cli` → `KEYS *celery*`
3. Check worker logs for "Received task" message

### If you see "internal server error":

1. Check FastAPI logs for exceptions
2. Check Redis connection: `redis-cli ping` should return PONG
3. Verify `REDIS_URL` in `.env`: should be `redis://localhost:6379/0`

### If result is always "failed":

1. Check worker error logs
2. Ensure audio file downloads successfully from S3
3. Verify ML models are loaded correctly
4. Check S3 credentials in `.env`

### If you see TypeError with AsyncResult:

1. Restart FastAPI after modifying imports
2. Check `from celery.result import AsyncResult` is present
3. Verify no duplicate endpoint definitions

---

## Production Recommendations

### 1. **Use Connection Pooling**

```python
# In celery_app.py
celery.conf.broker_pool_limit = 1
celery.conf.broker_connection_retry_on_startup = True
```

### 2. **Monitor Task Queue**

```python
# Add metrics
from prometheus_client import Counter
task_received = Counter('celery_task_received_total', 'Total tasks received')
```

### 3. **Implement Task Timeouts**

```python
# Already done in tasks.py:
@celery.task(
    task_time_limit=3600,      # 1 hour hard limit
    task_soft_time_limit=3300   # 55 min soft warning
)
```

### 4. **Use Results Expiration**

```python
# Already done in celery_app.py:
result_expires=3600  # Results auto-delete after 1 hour
```

### 5. **Add Dead Letter Queue**

For failed tasks that exceed max retries - implement in production.

### 6. **Use Separate Redis Instance for Results**

```python
# In production, use separate Redis for result backend
celery.conf.result_backend = 'redis://redis-backend:6379/1'
celery.conf.broker_url = 'redis://redis-broker:6379/0'
```

---

## Summary of Changes

| File            | Issue                                | Fix                                                         |
| --------------- | ------------------------------------ | ----------------------------------------------------------- |
| `celery_app.py` | No task tracking config              | Added full conf with task_track_started=True, include tasks |
| `main.py`       | Duplicate broken endpoints           | Removed duplicate, added proper AsyncResult import          |
| `tasks.py`      | Exception returns instead of raising | Changed to raise, added retry config                        |
| `schemas.py`    | Missing error field                  | Added optional error field                                  |

---

## Key Learnings

1. **Celery must track task states** - `task_track_started=True` is essential
2. **Task discovery must be explicit** - Use `include` parameter
3. **Exceptions must be raised** - Returning error dicts doesn't mark tasks as FAILURE
4. **Redis must be the broker AND backend** - For reliable state tracking
5. **Serialization must be JSON** - For compatibility across systems
6. **Always validate result structure** - Before returning to clients

All issues should now be resolved. Your task states will properly transition: PENDING → STARTED → SUCCESS/FAILURE ✓
