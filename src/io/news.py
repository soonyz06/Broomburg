import datetime
import urllib.parse
import feedparser
import polars as pl
from concurrent.futures import ThreadPoolExecutor

def get_feed(url, symbol, source, is_sec):
    headers = {'User-Agent': 'FinancialAggregator/1.0 (contact@example.com)'} if is_sec else {}
    feed = feedparser.parse(url, request_headers=headers)
    rows = []
    for entry in feed.entries:
        try:
            dt = datetime.datetime.fromisoformat(entry.get("updated")).date() if is_sec else datetime.date(*entry.published_parsed[:3])
            rows.append({"date": dt, "symbol": symbol, "source": source, "headline": entry.get("title"), "link": entry.get("link")})
        except: continue
    return rows

def fetch_financial_data(symbols, limit=100, **kwargs):
    sources = ["reuters.com", "bloomberg.com", "cnbc.com", "wsj.com"]
    tasks = []
    for s in symbols:
        tasks.append((f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={s}&output=atom", s, "SEC", True))
        for src in sources:
            q = urllib.parse.quote(f'"{s}" site:{src}')
            tasks.append((f"https://news.google.com/rss/search?q={q}", s, src, False))
    with ThreadPoolExecutor(max_workers=10) as exe:
        results = list(exe.map(lambda p: get_feed(*p), tasks))
    flat = [item for sub in results for item in sub]
    return pl.DataFrame(flat).unique(subset=["headline", "symbol"]).sort("date", descending=True).head(limit) if flat else pl.DataFrame()
