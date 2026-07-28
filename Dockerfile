# syntax=docker/dockerfile:1.7

# Python 3.12: PyTorch publishes CPU wheels for it (3.14 has none yet).
ARG PYTHON_VERSION=3.12
ARG MODEL_ID=Salesforce/blip-image-captioning-large
# float16 halves the on-disk checkpoint (1.8GB -> ~0.9GB). Weights are upcast
# back to float32 at load time, so CPU inference quality is unchanged.
ARG MODEL_DTYPE=float16

# ---------------------------------------------------------------- base ------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# ------------------------------------------------------------- builder ------
# Dependencies only. Isolated so app-code edits never invalidate the pip layer.
FROM base AS builder

RUN python -m venv "$VIRTUAL_ENV"

# torch first, from the CPU-only index. The default PyPI wheel drags in ~2.5GB
# of bundled CUDA libraries that are dead weight on a CPU-inference image.
# --no-compile: .pyc files across the venv weigh ~210MB. Skipping them trades a
# few seconds of first-import parsing (once, at --preload) for that space.
ARG TORCH_SPEC=torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --index-url https://download.pytorch.org/whl/cpu "${TORCH_SPEC}"

# Everything else from PyPI. torch is already satisfied, so it is not re-fetched.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile -r requirements.txt

# ---------------------------------------------------------------- model ------
# Re-export the checkpoint into a plain directory. Compared to shipping the raw
# HF cache this drops the blobs/snapshots/refs scaffolding and, at float16,
# halves the weights.
FROM builder AS model
ARG MODEL_ID
ARG MODEL_DTYPE
ENV HF_HOME=/tmp/hf
RUN --mount=type=cache,target=/tmp/hf \
    MODEL_ID="${MODEL_ID}" MODEL_DTYPE="${MODEL_DTYPE}" python - <<'PY'
import os, torch
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = os.environ["MODEL_ID"]
dtype = getattr(torch, os.environ["MODEL_DTYPE"])

processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(dtype).eval()

processor.save_pretrained("/opt/model")
model.save_pretrained("/opt/model", safe_serialization=True)
PY

# ---------------------------------------------------------------- prune ------
# Strip build-time-only and test-only payload from the venv. Kept separate from
# the model stage so both run in parallel and neither invalidates the other.
FROM builder AS pruned
RUN set -eux; \
    SP="$(ls -d /opt/venv/lib/python*/site-packages)"; \
    # torch ships C++ headers and its own test suite: unused for inference
    rm -rf "$SP"/torch/test "$SP"/torch/include "$SP"/torch/utils/benchmark; \
    # torch/bin is gtest binaries EXCEPT torch_shm_manager, which torch needs at import
    find "$SP"/torch/bin -type f ! -name 'torch_shm_manager' -delete; \
    # huggingface_hub CLI extras -- nothing serves HTTP-free inference
    pip uninstall -y --quiet hf_xet rich typer pygments markdown-it-py shellingham mdurl 2>/dev/null || true; \
    # packaging tooling is not needed once the venv is built
    pip uninstall -y --quiet pip setuptools wheel 2>/dev/null || true; \
    rm -rf "$SP"/pip "$SP"/setuptools "$SP"/wheel "$SP"/pkg_resources "$SP"/*.dist-info/RECORD; \
    find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    find /opt/venv -name '*.pyc' -delete; \
    find /opt/venv \( -name 'tests' -o -name 'test' \) -type d -prune -exec rm -rf {} + ; \
    du -sh /opt/venv

# ------------------------------------------------------------- runtime ------
FROM base AS runtime

ENV MODEL_ID=/opt/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PORT=5000 \
    HOME=/home/app

# Non-root. Fixed uid/gid so mounted volumes have predictable ownership.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app

COPY --from=pruned --chown=10001:10001 /opt/venv  /opt/venv
COPY --from=model  --chown=10001:10001 /opt/model /opt/model

WORKDIR /app
COPY --chown=10001:10001 app.py wsgi.py ./
COPY --chown=10001:10001 templates/ ./templates/

USER 10001:10001
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/\", timeout=4)"

# 1 worker by default: BLIP-large is ~1.9GB resident, so each extra worker costs
# that much RAM. --preload loads the model before forking, letting additional
# workers (WEB_CONCURRENCY=N) share those pages copy-on-write.
# --timeout 120 because CPU captioning easily exceeds gunicorn's 30s default.
CMD ["sh", "-c", "exec gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers ${WEB_CONCURRENCY:-1} \
    --threads ${GUNICORN_THREADS:-4} \
    --worker-class gthread \
    --timeout ${GUNICORN_TIMEOUT:-120} \
    --graceful-timeout 30 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    wsgi:app"]
