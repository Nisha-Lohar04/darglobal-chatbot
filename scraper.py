"""Polite scraper for public DarGlobal and Wasalt pages.

Usage:  python scraper.py            # writes data/corpus.json
Env:    MAX_PAGES_PER_SITE (default 60), DELAY_SECONDS (default 1.0),
        SCRAPER_CONTACT (optional contact string added to the User-Agent)

Respects robots.txt, stays on the seed domains, and only reads public pages.
Wasalt is a JS-heavy site, so besides visible text we also read embedded
__NEXT_DATA__ JSON and the public WordPress API of blog.wasalt.sa.
"""
import json, os, re, time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

_contact = os.getenv("SCRAPER_CONTACT", "")
UA = "Mozilla/5.0 (compatible; AssignmentBot/1.0" + (f"; {_contact}" if _contact else "") + ")"
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_SITE", "60"))
DELAY = float(os.getenv("DELAY_SECONDS", "1.0"))
OUT = os.path.join(os.path.dirname(__file__), "data", "corpus.json")

SITES = {
    "DarGlobal": {
        "seeds": ["https://darglobal.co.uk/", "https://darglobal.co.uk/press",
            "https://darglobal.co.uk/insights",
            "https://darglobal.co.uk/press/darglobal-partners-with-wasalt-to-advertise-trump-tower-jeddah",
            "https://darglobal.co.uk/about"],
        "domains": {"darglobal.co.uk", "www.darglobal.co.uk"},
    },
    "Wasalt": {
        "seeds": ["https://wasalt.sa/en", "https://wasalt.sa/en/corporate",
                  "https://wasalt.com/en/about", "https://wasalt.sa/en/corporate", "https://wasalt.com/en/about",],
        "domains": {"wasalt.sa", "www.wasalt.sa", "wasalt.com", "www.wasalt.com"},
    },
}
SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp|pdf|zip|mp4|css|js|ico|woff2?)(\?|$)", re.I)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept-Language": "en"})


def robots_for(url):
    p = urlparse(url)
    rp = RobotFileParser()
    try:
        rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
    except Exception:
        return None
    return rp


def clean_text(soup):
    for t in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        t.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if len(l) > 2)


def next_data_text(soup):
    """Pull readable strings out of Next.js embedded JSON (for JS-rendered pages)."""
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return ""
    try:
        data = json.loads(tag.string)
    except Exception:
        return ""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("title", "name", "description", "content", "text", "summary",
                         "city", "district", "price", "type"):
                    if isinstance(v, (str, int, float)) and str(v).strip():
                        out.append(f"{k}: {v}")
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(data.get("props", {}))
    return "\n".join(dict.fromkeys(out))[:8000]


def sitemap_urls(site_root, domains):
    urls = []
    for path in ("/sitemap.xml", "/sitemap_index.xml"):
        try:
            r = SESSION.get(site_root.rstrip("/") + path, timeout=20)
            if r.ok and "<loc>" in r.text:
                urls += re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text)
        except Exception:
            pass
    return [u for u in urls if urlparse(u).netloc in domains and not u.endswith(".xml")]


def crawl(name, cfg):
    docs, seen = [], set()
    queue = deque(cfg["seeds"])
    for root in {f"https://{urlparse(s).netloc}" for s in cfg["seeds"]}:
        queue.extend(sitemap_urls(root, cfg["domains"])[:MAX_PAGES])
    robots = {}
    while queue and len(docs) < MAX_PAGES:
        url = urldefrag(queue.popleft())[0].rstrip("/")
        if url in seen or SKIP_EXT.search(url):
            continue
        seen.add(url)
        host = urlparse(url).netloc
        if host not in cfg["domains"]:
            continue
        if host not in robots:
            robots[host] = robots_for(url)
        if robots[host] and not robots[host].can_fetch(UA, url):
            print("  robots.txt disallows", url)
            continue
        try:
            r = SESSION.get(url, timeout=25)
            time.sleep(DELAY)
        except Exception as e:
            print("  fail", url, e)
            continue
        if not r.ok or "text/html" not in r.headers.get("content-type", ""):
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else url
        extra = next_data_text(soup)
        for a in soup.find_all("a", href=True):
            link = urldefrag(urljoin(url, a["href"]))[0]
            if urlparse(link).netloc in cfg["domains"] and link.rstrip("/") not in seen:
                queue.append(link)
        text = clean_text(soup)
        if extra:
            text += "\n" + extra
        if len(text) > 200:
            docs.append({"source": name, "url": url, "title": title, "text": text})
            print(f"  [{name}] {len(docs):>3} {url}")
    return docs


def wasalt_blog(limit=30):
    docs = []
    try:
        r = SESSION.get("https://blog.wasalt.sa/en/wp-json/wp/v2/posts",
                        params={"per_page": limit, "_fields": "link,title,content"}, timeout=25)
        for p in r.json():
            text = BeautifulSoup(p["content"]["rendered"], "html.parser").get_text("\n")
            text = re.sub(r"\n{2,}", "\n", text).strip()
            if len(text) > 200:
                docs.append({"source": "Wasalt", "url": p["link"],
                             "title": BeautifulSoup(p["title"]["rendered"], "html.parser").get_text(),
                             "text": text})
        print(f"  [Wasalt blog] {len(docs)} posts")
    except Exception as e:
        print("  blog API failed:", e)
    return docs


def main():
    corpus = []
    for name, cfg in SITES.items():
        print("Crawling", name)
        corpus += crawl(name, cfg)
    corpus += wasalt_blog()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(corpus)} documents to {OUT}")


if __name__ == "__main__":
    main()
