# VibeAI-MS1 Commands & Links

## Start the stack
```powershell
cd c:\dev\SAAS\vibeai-ms1
docker compose up --build
```

## Start only one service
```powershell
cd c:\dev\SAAS\vibeai-ms1
docker compose up --build api
```

```powershell
cd c:\dev\SAAS\vibeai-ms1
docker compose up --build worker
```

## Service URLs
- FastAPI API: `http://localhost:8000`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- MLflow: `http://localhost:5000`
- Redis: `redis://localhost:6379`

## API health and test commands
```powershell
curl http://localhost:8000
curl http://localhost:8000/health
curl -X POST -F "file=@audio.mp3" http://localhost:8000/api/v1/analyze
curl -s http://localhost:8000/api/v1/result/{task_id} | python -m json.tool
```

## Docker compose helpers
```powershell
cd c:\dev\SAAS\vibeai-ms1
# Start the stack in the foreground (rebuild only when needed)
docker compose up

# Start the stack in detached mode
docker compose up -d

# Rebuild only when you've changed code or dependencies
docker compose up --build

# Rebuild only the API and worker services
docker compose build api worker

# Stop all services gracefully
docker compose stop

# Restart services
docker compose restart

# Shut down and remove containers, networks, and default volumes
docker compose down

# View logs for all services
docker compose logs --tail=100 --follow

# View logs for a specific service
docker compose logs --tail=100 --follow api

docker compose logs --tail=100 --follow worker

# List running containers
docker compose ps

# Validate compose YAML
docker compose config
```

## Celery
```powershell
cd c:\dev\SAAS\vibeai-ms1
celery -A app.workers.celery_app worker --pool=solo --loglevel=info
```

```powershell
celery -A app.workers.celery_app inspect active
celery -A app.workers.celery_app inspect registered
celery -A app.workers.celery_app inspect stats
celery -A app.workers.celery_app purge
```

## Redis / Windows troubleshooting
```powershell
netstat -ano | findstr :8000
taskkill /PID {PID} /F
Get-Process | findstr redis
```

## Optional debug checks
```powershell
python -c "from app.core.config import settings; print(settings.redis_url, settings.celery_result_backend, settings.s3_base_url)"
```

## Notes
- The FastAPI service is served by `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- The worker uses `celery -A app.workers.celery_app worker --pool=solo --loglevel=info`.
- If the compose stack fails, run `docker compose config` first to validate the YAML.
- Use `docker compose up --build` after any code or dependency update.
