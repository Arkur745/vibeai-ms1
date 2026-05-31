FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------------------------------
# Create Required Directories
# ---------------------------------

RUN mkdir -p /app/temp_uploads && \
    chmod 777 /app/temp_uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]