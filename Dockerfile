FROM nvcr.io/nvidia/pytorch:23.09-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
# ENV HF_HOME /app/.cache/huggingface
# RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
#                AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3'); \
#                AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')"
EXPOSE 11435
# For GPU, usually 1 worker per GPU is optimal. For CPU, you can have more.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "11435"]