FROM python:3.11-slim

# System dependencies for audio processing & build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY . .

# Expose FastAPI default port
EXPOSE 7860

# Run FastAPI server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
