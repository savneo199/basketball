# Base image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps 
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . /app

# Cloud Run will inject PORT (e.g., 8080). Streamlit must bind to 0.0.0.0 and that PORT.
# Do NOT hardcode a port; pass via CLI so $PORT is used at runtime.
EXPOSE 8080

# Optional but useful for Streamlit on Cloud Run
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check (short timeout so it doesn't hang)
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8080}/ || exit 1

# Use a shell so $PORT expands
CMD ["bash", "-lc", "streamlit run app/app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]
