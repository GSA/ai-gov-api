# ---- Builder Stage ----
# This is uv's official image: Python 3.11 and uv pre-installed.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
LABEL stage=builder

WORKDIR /opt/project
COPY pyproject.toml uv.lock* ./
RUN uv --version
RUN uv venv /opt/venv --python 3.11

RUN . /opt/venv/bin/activate && \
    uv sync --frozen --no-dev --no-cache --active

COPY . .

# ---- Runtime Stage ----
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /opt/project

RUN groupadd --system appuser && useradd --system --gid appuser appuser

# Bring in the dependencies.
COPY --chown=appuser:appuser --from=builder /opt/venv /opt/venv
# Copy app code from the builder stage.
# We are pulling in everything not in dockerignore to 
# give us easier access to alembic and script
COPY --chown=appuser:appuser --from=builder /opt/project/ ./


USER appuser

# Ensures Python output is sent to terminal for logging.
ENV PYTHONUNBUFFERED=1
# To allow executables installed in the venv (like uvicorn) be directly callable.
ENV PATH="/opt/venv/bin:$PATH"
# needed for `uvicorn app.main:app` if WORKDIR is /opt/project and code is in /opt/project/app/.
ENV PYTHONPATH="/opt/project"

# Expose the port your application will run on.
EXPOSE 8080

# Command to run the application.
# Uvicorn will be found via the PATH updated to include /opt/venv/bin.
# Adjust `app.main:app` and `--workers` as needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "3"]