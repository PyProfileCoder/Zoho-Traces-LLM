FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --use-deprecated=legacy-resolver -r requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads vector_data

# Expose ports
EXPOSE 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/api/health || exit 1

# Run the application
CMD ["python", "run_api_server.py"]
