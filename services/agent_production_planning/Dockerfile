# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Install curl for health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Create a non-root user early and set up permissions
RUN useradd -m appuser

# Install the project's dependencies using the lockfile and settings (as root)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Add the rest of the project source code
ADD . /app

# Copy data file and HTML interface
COPY mydata.xlsx /app/mydata.xlsx
COPY upload.html /app/upload.html

# Switch to appuser and run the final sync to set up the virtual environment
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set file path for data file
ENV file_path="/app/mydata.xlsx"

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Make server.py executable
RUN chmod +x /app/server.py

# Add basic health check for the container
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/ || exit 1

# Reset the entrypoint, don't invoke `uv` by default
ENTRYPOINT []

# Command to run when the container starts - now using server.py
CMD ["uv", "run", "python", "/app/server.py"]