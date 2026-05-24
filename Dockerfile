# ---------------------------------
# Base Image
# ---------------------------------
FROM python:3.10-slim

# ---------------------------------
# System Dependencies
# ---------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------
# Working Directory
# ---------------------------------
WORKDIR /app

# ---------------------------------
# Copy Requirements
# ---------------------------------
COPY requirements.txt .

# ---------------------------------
# Install Python Dependencies
# ---------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------
# Copy Project Files
# ---------------------------------
COPY . .

# ---------------------------------
# Expose Port
# ---------------------------------
EXPOSE 8000

# ---------------------------------
# Start FastAPI Server
# ---------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]