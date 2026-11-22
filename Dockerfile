FROM python:3.11-slim

# Copy the uv binary from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevent Python bytecode files and enable real-time logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Create a non-root user for production security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# 1. Copy only lockfiles first (caches dependency layer unless packages change)
COPY pyproject.toml uv.lock ./

# 2. Install exact dependencies into a virtual environment (.venv)
RUN uv sync --frozen --no-install-project --no-dev

# 3. Copy application source code and set ownership
COPY --chown=appuser:appgroup . .

# Add the virtual environment to PATH so `python` uses it automatically
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

# Change "src/main.py" to your actual entry file
CMD ["python", "src/main.py"]
