FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Hugging Face Spaces runs containers as UID 1000 and expects port 7860.
RUN useradd -m -u 1000 user && chown -R user /app
USER user

ENV PORT=7860
EXPOSE 7860

# data/corpus.json is committed; if missing, app.py scrapes on first start.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
