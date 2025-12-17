FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
# Note: We install dependencies first to leverage caching
RUN pip install --no-cache-dir .[dev]

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python3", "-m", "pytest", "tests/"]
