# Stage 1: Builder
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

# Set Python to unbuffered mode
ENV PYTHONUNBUFFERED=1

# Update apt and install build essentials, python3, and python3-venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt into the working directory
COPY requirements.txt .

# Create a Python virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment for subsequent commands in this stage
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies from requirements.txt into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set Python to unbuffered mode for the final image
ENV PYTHONUNBUFFERED=1

# Set working directory for the final image
WORKDIR /app

# Update apt and install necessary runtime dependencies for python3 in the final image.
# libexpat1 provides libexpat.so.1, which is a common dependency for Python.
# python3-distutils is often needed for various Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages (site-packages) from the builder stage's virtual environment
# to a location in the final image. We'll then add this location to PYTHONPATH.
COPY --from=builder /opt/venv/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the executables (like uvicorn, which is a script) from the builder stage's virtual environment's bin directory
# to a location in the final image's PATH. We will not rely on the shebang of these scripts.
COPY --from=builder /opt/venv/bin/ /usr/local/bin/

# Set PYTHONPATH to include the directory where the site-packages were copied.
# This ensures that the Python interpreter can find the installed modules.
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:$PYTHONPATH

# Copy your application code into the final image
COPY app/ app/
COPY .gitignore .
COPY README.md .

# Expose port 8000 for the Uvicorn application
EXPOSE 8000

# Command to run the application.
# We explicitly tell the 'python3' interpreter to run the 'uvicorn' module.
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
