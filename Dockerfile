# Streamlit demo app for the finish-time predictor.
#
#   docker build -t pacebrain .
#   docker run --rm -p 8501:8501 pacebrain
#
# Then open http://localhost:8501.

FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE: .pyc files are dead weight in an image layer.
# PYTHONUNBUFFERED: without it, Python buffers stdout when it is a pipe, so
# `docker logs` shows nothing until the buffer flushes — which for a
# long-running server can be never.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before source, so editing a .py file does not invalidate the
# pip layer. That layer is the expensive one — torch alone is most of it.
COPY requirements.txt ./

# CPU-only torch, explicitly. The default PyPI wheel bundles the CUDA
# runtime and is roughly 2 GB larger, which is pure waste here: this app runs
# a 2,561-parameter MLP on one row at a time and would not benefit from a GPU
# even if the container had one.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2.0" \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY pyproject.toml README.md ./

# app.py and config.py resolve paths relative to the working directory, so
# src/ has to be importable without an editable install rewriting anything.
ENV PYTHONPATH=/app/src

# Train during the build so `docker run` is immediately useful.
#
# The alternative is shipping an image whose only behaviour is telling you to
# train a model first. Training is seeded and takes well under a minute on
# CPU, and it happens after the dependency layer, so iterating on source does
# not re-download torch.
#
# To use your own checkpoint instead, mount over it at run time:
#   docker run --rm -p 8501:8501 -v "$PWD/models:/app/models" pacebrain
RUN python src/pacebrain/train_finish.py

# Streamlit binds 127.0.0.1 by default, which is unreachable from outside the
# container. headless=true suppresses the first-run email prompt, which would
# otherwise block startup waiting on stdin that is not there.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run as a non-root user. Nothing here needs root, and the default in a
# container is root unless you say otherwise.
RUN useradd --create-home --uid 10001 pacebrain \
 && chown -R pacebrain:pacebrain /app
USER pacebrain

EXPOSE 8501

# Streamlit's own health endpoint, so an orchestrator can tell "process is
# up" apart from "app is actually serving".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/app.py"]
