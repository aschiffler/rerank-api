FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Create a virtual environment
RUN python3 -m venv /opt/venv
# Activate the virtual environment and install dependencies
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /opt/venv/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /opt/venv/bin/ /usr/local/bin/
# If Python needs to be explicitly copied:
#COPY --from=builder /usr/bin/python3.11 /usr/bin/python3.11
#COPY --from=builder /usr/bin/python3 /usr/bin/python3

COPY app/ app/
COPY .gitignore .
COPY README.md .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]