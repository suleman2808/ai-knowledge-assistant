# Container image for the assistant.
#
# One image, any host. Hugging Face Spaces serves on a fixed port 7860;
# Render, Railway, Fly and Cloud Run inject a PORT variable and kill a
# container that ignores it. docker-entrypoint.sh reconciles the two.
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

# The default port when a host does not specify one.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860
EXPOSE 7860

# A container that reports its own health, so an orchestrator can tell
# "starting" from "broken" without guessing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; port=os.environ.get('PORT') or os.environ.get('APP_PORT','7860'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=4).status==200 else 1)"

CMD ["./docker-entrypoint.sh"]
