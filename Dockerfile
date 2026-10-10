# ============================================
# 🌾 UZHAVAN AI - Dockerfile
# Multi-stage build for FastAPI + React frontend
# ============================================

# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY rag/ ./rag/
COPY agriculture/ ./agriculture/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data_collection/ ./data_collection/

# Copy built frontend static files
COPY --from=frontend-builder /app/frontend/dist ./frontend_dist

# Create storage directories
RUN mkdir -p models datasets temp_uploads

# Expose ports
EXPOSE 8000 5001

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
