# Quick Debugging Commands

## Redis CLI Commands

```bash
# Connect to Redis
redis-cli

# Check if Redis is running
ping
# Returns: PONG

# List all Celery task keys
KEYS *celery*

# Get task metadata
GET celery-task-meta-{task_id}

# Check Redis memory usage
INFO memory

# Monitor real-time commands
MONITOR

# Flush all (DANGER - development only)
FLUSHALL
```

## Celery Commands

```bash
# Check active workers
celery -A app.workers.tasks inspect active

# Check registered tasks
celery -A app.workers.tasks inspect registered

# Check stats
celery -A app.workers.tasks inspect stats

# Purge all pending tasks
celery -A app.workers.tasks purge
# Confirm: y

# Monitor real-time events
celery -A app.workers.tasks events
```

## FastAPI Testing

```bash
# Check if API is running
curl http://localhost:8000

# Health check
curl http://localhost:8000/health

# Submit task
curl -X POST -F "file=@audio.mp3" http://localhost:8000/api/v1/analyze

# Poll task result (replace {task_id} with actual ID)
curl http://localhost:8000/api/v1/result/{task_id}

# Pretty print JSON response
curl -s http://localhost:8000/api/v1/result/{task_id} | python -m json.tool
```

## Windows-Specific

```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process on port 8000 (replace PID)
taskkill /PID {PID} /F

# List Redis processes
Get-Process | findstr redis
```

## Celery Worker Startup (Windows)

```bash
# Start worker with verbose logging
celery -A app.workers.tasks worker --loglevel=debug --pool=solo

# Start with concurrency=1 (safer on Windows)
celery -A app.workers.tasks worker --concurrency=1 --pool=solo

# Start with task events (for monitoring)
celery -A app.workers.tasks worker --loglevel=info -E --pool=solo
```

## Logging

```bash
# Tail Celery logs in real-time
tail -f celery_worker.log

# Filter for specific task
grep -i "task_id" celery_worker.log

# Check error logs
grep -i "error\|failed\|exception" celery_worker.log
```

## Verify Configuration

```python
# In Python shell
from app.workers.celery_app import celery

# Check broker URL
print(celery.conf.broker_url)

# Check result backend URL
print(celery.conf.result_backend)

# Check if task_track_started is True
print(celery.conf.task_track_started)

# Check included modules
print(celery.conf.include)

# Check all config
import pprint
pprint.pprint(dict(celery.conf))
```

## Common Issues & Fixes

### Issue: "No active workers"

```bash
# Start worker
celery -A app.workers.tasks worker --pool=solo --loglevel=info
```

### Issue: "Redis connection refused"

```bash
# Start Redis
redis-server

# Or on Windows with WSL
wsl redis-server
```

### Issue: "Task not found"

```bash
# Ensure tasks are included in celery app
celery -A app.workers.tasks inspect registered

# If empty, check celery_app.py has:
# celery.conf.include = ['app.workers.tasks']
```

### Issue: "PENDING forever"

```bash
# Check if task_track_started is enabled
from app.workers.celery_app import celery
print(celery.conf.task_track_started)
# Should print: True

# Check Redis has the task key
redis-cli GET celery-task-meta-{task_id}
```

## Performance Monitoring

```bash
# Monitor task queue depth
redis-cli LLEN celery

# Monitor active tasks
celery -A app.workers.tasks inspect active

# Get worker stats
celery -A app.workers.tasks inspect stats

# Monitor Celery events (real-time)
celery -A app.workers.tasks events --dump
```

## Reset Everything (Development Only)

```bash
# Stop Celery worker (Ctrl+C in terminal)

# Purge Redis
redis-cli FLUSHALL

# Restart Redis
redis-server

# Restart worker
celery -A app.workers.tasks worker --pool=solo --loglevel=info

# Restart FastAPI
uvicorn app.main:app --reload
```
