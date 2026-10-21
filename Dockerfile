# Pinned by digest: the tag is republished upstream with each patch and security
# rebuild (the last one 11 hours before a slow deploy), and each
# new digest made every layer below rebuild, PyTorch's 1.6 GB included. Bump
# it deliberately, with the dependency pins (docker buildx imagetools inspect).
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

# uv pinned to the exact version that produced uv.lock — the same one CI installs.
COPY --from=ghcr.io/astral-sh/uv:0.9.21@sha256:15f68a476b768083505fe1dbfcc998344d0135f0ca1b8465c4760b323904f05a /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependency layer first, protected from source changes. When uv.lock does
# change, the download cache (kept by BuildKit between builds, never in the
# image) means only the packages that changed are fetched again.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# The embedding model, downloaded once, at build time, into the image: the
# running application never reaches the network for it, and so can never send
# a document or a question anywhere. Outside /app, so the chown below does not
# copy its weights into a second layer. It must be the model that EMBEDDING_MODEL
# in .env names: offline, any other model fails to load (see .env.example).
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV HF_HOME=/opt/models
RUN python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1], device='cpu')" "${EMBEDDING_MODEL}" \
    && chmod -R a+rX /opt/models
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app/ ./app/
ENV PYTHONPATH=/app/app

# Non-root: a compromised app is not root in the container.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Volatile args LAST so a new version does not invalidate the dependency layer.
ARG BUILD_TIME=dev
ARG APP_VERSION=dev
ARG APP_COMMIT=dev
ENV BUILD_TIME=${BUILD_TIME} \
    APP_VERSION=${APP_VERSION} \
    APP_COMMIT=${APP_COMMIT}

CMD ["uvicorn", "rag_my_docs.main:app", "--host", "0.0.0.0", "--port", "8000"]
