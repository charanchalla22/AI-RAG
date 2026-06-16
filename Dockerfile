FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages first to use Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

EXPOSE 8000

# Start the application using the correct folder structure path
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
