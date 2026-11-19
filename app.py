"""FastAPI RAG chatbot over scraped DarGlobal + Wasalt data, using a free OpenRouter model."""
import json, os, re
from typing import List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(BASE, "data", "corpus.json")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Free model IDs on OpenRouter change over time; set OPENROUTER_MODELS (comma separated)
# to override. They are tried in order until one answers.
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "openrouter/free,meta-llama/llama-3.3-70b-instruct:free",
).split(",") if m.strip()]
TOP_K = int(os.getenv("TOP_K", "8"))
PAGE_CHARS = int(os.getenv("PAGE_CHARS", "14000"))     # how much of a matched page to send
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "18000"))   # total context budget (characters)

JUNK_URL = ("success", "login", "cancel", "cart", "payment", "privacy", "terms", "cookie")
QUESTION_WORDS = {"what", "tell", "about", "does", "with", "that", "this", "have", "which",
                  "where", "when", "whose", "give", "info", "information", "details", "please",
                  "show", "list", "much", "many", "from", "your", "there", "their", "them"}
BRAND_WORDS = {"darglobal", "global", "wasalt"}        # appear in almost every page; no signal

app = FastAPI(title="DarGlobal & Wasalt Assistant")


def dedupe_lines(text: str) -> str:
    seen, out = set(), []
    for line in text.splitlines():
        key = line.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(line.strip())
    return "\n".join(out)


def load_docs():
    if not os.path.exists(CORPUS_PATH):
        import scraper
        scraper.main()
    docs = json.load(open(CORPUS_PATH, encoding="utf-8"))
    clean = []
    for d in docs:
        if any(j in d["url"].lower() for j in JUNK_URL):
            continue
        d["text"] = dedupe_lines(d["text"])
        if len(d["text"]) > 150:
            clean.append(d)
    return clean


def make_chunks(docs, size=1000, overlap=150):
    chunks = []
    for d in docs:
        t = d["text"]
        for i in range(0, len(t), size - overlap):
            piece = t[i:i + size].strip()
            if len(piece) > 120:
                chunks.append({"source": d["source"], "url": d["url"], "title": d["title"], "text": piece})
    return chunks


DOCS = load_docs()
CHUNKS = make_chunks(DOCS)
VEC = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
MATRIX = VEC.fit_transform([c["title"] + " " + c["text"] for c in CHUNKS]) if CHUNKS else None


def query_words(q: str):
    return [w for w in re.findall(r"[a-z]{3,}", q.lower())
            if w not in QUESTION_WORDS and w not in BRAND_WORDS and w not in ("the", "and", "for", "are")]


def matching_pages(q: str, limit=2):
    """Pages whose URL/title contain several words from the question (e.g. a project page)."""
    words = query_words(q)
    if not words:
        return []
    scored = []
    for d in DOCS:
        label = re.sub(r"[^a-z0-9]+", " ", (d["url"] + " " + d["title"]).lower())
        hits = sum(w in label for w in words)
        need = 1 if len(words) == 1 else 2
        if hits >= need:
            scored.append((hits, -len(d["url"]), d))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [d for _, _, d in scored[:limit]]


def retrieve(q: str, k: int = TOP_K):
    if MATRIX is None:
        return []
    scores = linear_kernel(VEC.transform([q]), MATRIX).ravel()
    words = query_words(q)
    for i, c in enumerate(CHUNKS):
        label = re.sub(r"[^a-z0-9]+", " ", (c["url"] + " " + c["title"]).lower())
        scores[i] += 0.15 * sum(w in label for w in words)
    ql = q.lower()
    both = "wasalt" in ql and bool(re.search(r"dar ?global|dar al ?arkan", ql))
    related = bool(re.search(r"relat|partner|owner|parent|subsidiar|affiliat|connect|link|group|sister", ql))
    if both or (related and ("wasalt" in ql or re.search(r"dar ?global", ql))):
        # Cross-company question: favour chunks that mention both sides of the link.
        for i, c in enumerate(CHUNKS):
            t = c["text"].lower()
            if "wasalt" in t and re.search(r"dar ?global|dar al ?arkan|quara", t):
                scores[i] += 0.6
    idx = scores.argsort()[::-1][:k * 3]
    picked, per_url = [], {}
    for i in idx:
        if scores[i] <= 0.02 or len(picked) >= k:
            continue
        u = CHUNKS[i]["url"]
        if per_url.get(u, 0) < 3:                      # max 3 chunks per page for variety
            per_url[u] = per_url.get(u, 0) + 1
            picked.append(CHUNKS[i])
    return picked


def build_context(q: str):
    """Full text of the best-matching page(s) first, then the top retrieved chunks."""
    parts, sources, used = [], [], set()
    for d in matching_pages(q):
        parts.append(f"({d['source']}) {d['title']} - {d['url']}\n{d['text'][:PAGE_CHARS]}")
        sources.append({"title": d["title"], "url": d["url"], "source": d["source"]})
        used.add(d["url"])
    for h in retrieve(q):
        if h["url"] in used and h["text"] in "\n".join(parts):
            continue
        parts.append(f"({h['source']}) {h['title']} - {h['url']}\n{h['text']}")
        if h["url"] not in {s["url"] for s in sources}:
            sources.append({"title": h["title"], "url": h["url"], "source": h["source"]})
    context, total = [], 0
    for p in parts:
        if total + len(p) > MAX_CONTEXT:
            break
        context.append(p)
        total += len(p)
    return "\n\n---\n\n".join(context), sources


SYSTEM = (
    "You are an assistant for DarGlobal (Dar Global, a London-listed luxury real estate "
    "developer) and Wasalt (a Saudi real estate platform). Answer ONLY from the context "
    "provided below, which is text scraped from the companies' public websites. "
    "The text may contain menu items and repeated fragments; ignore those and use the "
    "substantive facts. If the context does not contain the answer, say you don't have "
    "that information. Be concise and well organised (short paragraph or a few bullets), "
    "say which company a fact belongs to, and never invent prices, dates or figures. "
    "Never mention 'context' or numbered sources like [1]; just answer, including concrete "
    "details the text gives (location, number of floors, partners, dates, amenities, unit types)."
)


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str
    history: Optional[List[Msg]] = []


def call_openrouter(messages):
    if not API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY is not set on the server.")
    errors = []
    for model in MODELS:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "HTTP-Referer": os.getenv("APP_URL", "http://localhost"),
                         "X-Title": "DarGlobal Wasalt Chatbot"},
                json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1200},
                timeout=90,
            )
            if r.ok:
                content = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if content:
                    return content, model
                errors.append(f"{model}: empty reply")
            else:
                errors.append(f"{model}: {r.status_code} {r.text[:150]}")
        except Exception as e:
            errors.append(f"{model}: {e}")
    print("OpenRouter errors:", *errors, sep="\n  ")
    raise HTTPException(502, "All models failed. " + " || ".join(errors))


@app.post("/chat")
def chat(body: ChatIn):
    q = body.message.strip()
    if not q:
        raise HTTPException(400, "Empty message")
    context, sources = build_context(q)
    msgs = [{"role": "system", "content": SYSTEM + "\n\nContext:\n" + (context or "(nothing relevant found)")}]
    msgs += [m.model_dump() for m in (body.history or [])[-6:]]
    msgs.append({"role": "user", "content": q})
    answer, model = call_openrouter(msgs)
    return {"answer": answer, "sources": sources, "model": model}


@app.get("/debug")
def debug(q: str):
    """Shows exactly what text the model would receive. Handy for testing retrieval."""
    context, sources = build_context(q)
    return {"sources": sources, "context_chars": len(context), "context_preview": context[:3000]}


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(DOCS), "chunks": len(CHUNKS), "models": MODELS}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DarGlobal &amp; Wasalt Assistant</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--mut:#6b6a66;--acc:#8a6d3b;--line:#e4e1da}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--card:#211f1c;--ink:#f1efe9;--mut:#a09d95;--acc:#d3b377;--line:#34312c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:14px 16px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:18px}small{color:var(--mut)}
#log{flex:1;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:12px;max-width:820px;width:100%;margin:0 auto}
.m{padding:10px 14px;border-radius:12px;max-width:85%;white-space:pre-wrap;background:var(--card);border:1px solid var(--line)}
.u{align-self:flex-end;background:var(--acc);color:#fff;border:none}.src{font-size:13px;color:var(--mut);margin-top:6px}.src a{color:var(--acc)}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);max-width:820px;width:100%;margin:0 auto}
input{flex:1;padding:12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit}
button{padding:0 18px;border:none;border-radius:10px;background:var(--acc);color:#fff;font:inherit;cursor:pointer}
</style></head><body>
<header><h1>DarGlobal &amp; Wasalt Assistant</h1><small>Answers come from publicly scraped website content.</small></header>
<div id="log"><div class="m">Hi! Ask me about DarGlobal projects (e.g. Trump International Hotel &amp; Tower Dubai) or the Wasalt platform.</div></div>
<form id="f"><input id="q" placeholder="Ask a question..." autocomplete="off" required><button>Send</button></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),q=document.getElementById('q');let hist=[];
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=1e9;return d}
f.onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t)return;q.value='';add(t,'u');const w=add('Thinking...','');
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t,history:hist})});
const j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');w.textContent=j.answer;
if(j.sources&&j.sources.length){const s=document.createElement('div');s.className='src';s.append('Sources: ');
j.sources.slice(0,3).forEach((x,i)=>{const a=document.createElement('a');a.href=x.url;a.target='_blank';a.rel='noopener';a.textContent=x.source+': '+(x.title||x.url).slice(0,40);s.append(i?' | ':'',a)});w.appendChild(s)}
hist.push({role:'user',content:t},{role:'assistant',content:j.answer})}catch(err){w.textContent='Sorry, something went wrong: '+err.message}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE