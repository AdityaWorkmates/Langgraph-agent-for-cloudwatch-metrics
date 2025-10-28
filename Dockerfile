# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Install system dependencies and AWS CLI
RUN apt-get update && apt-get install -y \
    curl \
    unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set uv environment variables for better performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy lock and project metadata files
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev or project installation)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy the rest of the application code
COPY . .

# Optional: Ensure scripts/entry points use the installed venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 6000

# Set environment variables
ENV FLASK_ENV=production

# Reset entrypoint to not run uv by default
ENTRYPOINT []

# Run the application
CMD ["uv", "run", "api2.py"]