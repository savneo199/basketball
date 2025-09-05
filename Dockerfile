# ---- Base image ----
FROM python:3.11-slim

# Fast, clean Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (add more here if a lib complains during pip install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# ---- Install Python deps (cached layer) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy project ----
COPY . /app

# Streamlit must bind to 0.0.0.0 and use the PORT env (Cloud Run uses 8080)
ENV PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8080

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080')" || exit 1

# Update app path below if your entry file differs
# e.g. "streamlit run app/streamlit_app.py" or "streamlit run app/main.py"
CMD ["streamlit", "run", "app/app.py"]
