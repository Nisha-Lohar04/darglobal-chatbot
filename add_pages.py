"""Add specific public pages (and two curated notes) to data/corpus.json.

Usage:  python add_pages.py
Safe to re-run: entries are replaced by URL, never duplicated.

- Respects robots.txt (pages a site disallows are skipped, not worked around).
- If normal text cleaning leaves too little, tries fallbacks: whole-body text, then text
  embedded in script data (Next.js style pages), then meta descriptions.
- Appends two hand-written notes built from public press coverage, because Wasalt's own
  site disallows crawlers. They are labelled "Curated note" and cite their public sources.
"""
import json, os, re, time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import scraper

EXTRA = [
    ("DarGlobal", "https://darglobal.co.uk/press/darglobal-partners-with-wasalt-to-advertise-trump-tower-jeddah"),
    ("DarGlobal", "https://darglobal.co.uk/press"),
    ("DarGlobal", "https://darglobal.co.uk/about"),
]

CURATED = [
    {
        "source": "Wasalt",
        "url": "https://www.gulftech-news.com/en/?p=69769",
        "title": "Curated note: What is Wasalt (public press coverage)",
        "text": (
            "What is Wasalt? Wasalt is a Saudi Arabian real estate platform, available as a website and "
            "mobile app, where people buy, rent and list residential and commercial property, and where "
            "brokers, landlords and developers advertise. Wasalt is licensed by REGA (the General Real "
            "Estate Authority) as a real estate portal, and by Infath (the Entrustment and Liquidation "
            "Center) to organise electronic and hybrid real estate auctions. In January 2025 Wasalt "
            "reported more than 750,000 registered users and about 100,000 monthly leads, and launched "
            "AI tools for property descriptions, valuations and ROI prediction. Ziad El Chaar is "
            "Chairman of Wasalt. Sources: Gulf Tech News (January 2025), Zawya and TradeArabia press "
            "coverage of Wasalt's Infath licence and growth.\n"
            "This note was written by hand from public press coverage; Wasalt's own website disallows "
            "crawlers, so it was not scraped."
        ),
    },
    {
        "source": "DarGlobal",
        "url": "https://darglobal.co.uk/press/darglobal-partners-with-wasalt-to-advertise-trump-tower-jeddah",
        "title": "Curated note: How DarGlobal and Wasalt are related (public sources)",
        "text": (
            "How are DarGlobal and Wasalt related? The relationship between DarGlobal and Wasalt has "
            "two parts. 1) Common parent link: DarGlobal (Dar Global) is the international property "
            "arm of Saudi developer Dar Al Arkan (The National, June 2025). Wasalt is described on its "
            "corporate pages as a joint venture between Dar Al Arkan and Quara Holding, a technology "
            "startup incubator. So both companies are connected to Dar Al Arkan. Ziad El Chaar is CEO "
            "of DarGlobal and Chairman of Wasalt. 2) Commercial partnership: in April 2025 DarGlobal "
            "announced a partnership with Wasalt under which Trump Tower Jeddah is advertised and "
            "listed on Wasalt's website and app to reach investors and property seekers across Saudi "
            "Arabia (DarGlobal press release, 16 April 2025).\n"
            "This note was written by hand from public sources: the DarGlobal press release, The "
            "National, Gulf Tech News and Wasalt's public corporate description."
        ),
    },
]


def body_text_fallbacks(soup, html):
    """Try progressively looser ways to get readable text. Returns (method, text)."""
    # 1) normal cleaning
    s1 = BeautifulSoup(html, "html.parser")
    t = scraper.clean_text(s1)
    if len(t) >= 200:
        return "cleaned", t
    # 2) whole body, only dropping scripts and styles
    s2 = BeautifulSoup(html, "html.parser")
    for tag in s2(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [re.sub(r"\s+", " ", l).strip() for l in s2.get_text("\n").splitlines()]
    t = "\n".join(l for l in lines if len(l) > 2)
    if len(t) >= 200:
        return "whole-body", t
    # 3) strings embedded in script data (Next.js flight data, JSON blobs)
    strings = []
    for sc in soup.find_all("script"):
        raw = sc.string or ""
        for m in re.finditer(r'"((?:[^"\\]|\\.){60,})"', raw):
            try:
                s = json.loads('"' + m.group(1) + '"')
            except Exception:
                continue
            if " " in s and not s.startswith(("http", "{", "[", "$")) and s.count(" ") > 8:
                strings.append(re.sub(r"<[^>]+>", " ", s).strip())
    t = "\n".join(dict.fromkeys(strings))
    if len(t) >= 200:
        return "script-data", t
    # 4) meta descriptions
    metas = [m.get("content", "") for m in soup.find_all("meta")
             if m.get("name") in ("description",) or m.get("property") in ("og:description",)]
    t = "\n".join(dict.fromkeys(x for x in metas if x))
    return "meta", t


def main():
    path = scraper.OUT
    corpus = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    by_url = {d["url"].rstrip("/") + ("#note" if d["title"].startswith("Curated note") else ""): d
              for d in corpus}
    robots, added = {}, 0
    for source, url in EXTRA:
        host = urlparse(url).netloc
        if host not in robots:
            robots[host] = scraper.robots_for(url)
        if robots[host] and not robots[host].can_fetch(scraper.UA, url):
            print("robots.txt disallows (skipped)", url)
            continue
        try:
            r = scraper.SESSION.get(url, timeout=25)
            time.sleep(scraper.DELAY)
        except Exception as e:
            print("fail", url, e)
            continue
        if not r.ok:
            print(r.status_code, url)
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else url
        method, text = body_text_fallbacks(soup, r.text)
        print(f"   fetched {len(r.text)} bytes of HTML; method={method}; text={len(text)} chars  {url}")
        if len(text) < 150:
            print("   still too little text, skipped")
            continue
        by_url[url.rstrip("/")] = {"source": source, "url": url, "title": title, "text": text}
        added += 1
        print(f"ok  {len(text):>6} chars  {url}")
    for note in CURATED:
        by_url[note["url"].rstrip("/") + "#note"] = note
        print("ok  curated note:", note["title"])
    json.dump(list(by_url.values()), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Added/updated {added} scraped pages + {len(CURATED)} curated notes; corpus now has {len(by_url)} documents.")


if __name__ == "__main__":
    main()