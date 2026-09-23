# Container image for the assistant.
#
# Built for Hugging Face Spaces (free CPU tier), which serves on port
# 7860 and runs the container as uid 1000. It is an ordinary image
# otherwise and runs anywhere Docker does.
#
# The knowledge base is built at image build time rather than on boot:
# ingestion needs no API key, only the local embedding model, so baking
# it in means the container starts ready to answer instead of spending
# the first request downloading a model.

FROM python:3.12-slim

# Hugging Face Spaces runs as uid 1000. Creating that user here means the
# application never needs to write to a root-owned directory.
RUN useradd --create-home --uid 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/app/src

# Dependencies first, so a code change does not reinstall them.
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app:app . .

# Build the vector store into the image. Also warms the ONNX model cache,
# so the first request does not pay for a 79 MB download.
RUN python -m scripts.ingest --quiet

# Spaces expects 7860; APP_PORT keeps the local default of 8000 intact.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860
EXPOSE 7860

# A container that reports its own health, so an orchestrator can tell
# "starting" from "broken" without guessing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
