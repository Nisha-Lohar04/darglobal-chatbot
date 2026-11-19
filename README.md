---
title: DarGlobal Wasalt Chatbot
emoji: 🏙️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# DarGlobal & Wasalt Chatbot

RAG chatbot over publicly scraped DarGlobal and Wasalt content, answered by a free OpenRouter model.

**Stack:** Python, requests + BeautifulSoup (scraping), TF-IDF retrieval (scikit-learn), FastAPI, OpenRouter, Docker.

## How it works
1. `scraper.py` crawls public pages (robots.txt respected, polite delay) from darglobal.co.uk, wasalt.sa, wasalt.com and the Wasalt blog's public WordPress API, and saves `data/corpus.json`.
2. `app.py` chunks the corpus, builds a TF-IDF index at startup, retrieves the top 5 chunks per question and sends them as context to OpenRouter. The prompt tells the model to answer only from context.
3. Answers include source links. `/health` shows chunk count and models.

## Run locally
```bash
pip install -r requirements.txt
python scraper.py                      # creates data/corpus.json
export OPENROUTER_API_KEY=sk-or-...    # free key from openrouter.ai/keys
uvicorn app:app --port 7860            # open http://localhost:7860
```

## Docker
```bash
docker build -t darglobal-bot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... darglobal-bot
```

## Deploy (Hugging Face Spaces, free, gives a public URL)
1. Create a Space at huggingface.co/new-space, SDK = **Docker**, blank template, public.
2. Space **Settings > Variables and secrets > New secret**: `OPENROUTER_API_KEY` = your key.
3. Push this folder (including `data/corpus.json`) to the Space repo:
   ```bash
   git init && git add . && git commit -m "init"
   git remote add space https://huggingface.co/spaces/<your-username>/darglobal-chatbot
   git push space HEAD:main
   ```
4. Wait for the build. Your URL is `https://<your-username>-darglobal-chatbot.hf.space`.

Alternative: Render.com > New Web Service > connect GitHub repo > Runtime Docker > add the env var.

## Notes
- Free OpenRouter model IDs change; override with `OPENROUTER_MODELS` (comma separated, tried in order).
- Never commit your API key. Use the host's secret store.
