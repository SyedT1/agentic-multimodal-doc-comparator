FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as UID 1000. Both src/data and .cache must be writable.
RUN useradd --create-home --uid 1000 appuser \
    && chown appuser:appuser /app

ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser docs/ ./docs/
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml

RUN pip3 install -r requirements.txt

USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.enableXsrfProtection=true", \
            "--server.enableCORS=true", \
            "--server.maxUploadSize=50"]
