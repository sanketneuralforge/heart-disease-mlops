# Slim base image matching your development Python version
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first — separate layer so Docker only re-installs
# when requirements.txt actually changes, not on every code edit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the serving API needs: app code, shared src code, and
# the pre-trained model artifact. The model is NOT trained inside this
# image — it must already exist in models/ (from running src/train.py
# locally, or pulling it from a CI run) before you build this image.
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user (basic container security practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Liveness check against the API's own /health endpoint.
# Uses Python's stdlib instead of curl, since the slim image doesn't
# include curl and installing it would bloat the image.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]