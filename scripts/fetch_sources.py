"""Fetch the text of every source in the De(ep)Composition bibliographies into data/source_text/ (one file per URL hash).

Usage: python scripts/fetch_sources.py
"""
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "source_text"


def key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def html_to_text(h: str) -> str:
    h = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    return re.sub(r"\s+", " ", unescape(h)).strip()


def fetch(url: str) -> tuple[str, int]:
    p = OUT / f"{key(url)}.txt"
    if p.exists():
        return url, len(p.read_text())
    try:
        r = subprocess.run(["curl", "-s", "-L", "-m", "25", "--max-filesize", "8000000", "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                            url], capture_output=True, timeout=40)
        raw = r.stdout
        if raw[:5] == b"%PDF-":
            tmp = OUT / f"{key(url)}.pdf"; tmp.write_bytes(raw)
            t = subprocess.run(["/Users/jessie/229-final-project/.venv/bin/python", "-c",
                                f"import pymupdf;d=pymupdf.open('{tmp}');print(' '.join(p.get_text() for p in d)[:400000])"], capture_output=True, text=True, timeout=120).stdout
            tmp.unlink(missing_ok=True)
        else:
            t = html_to_text(raw.decode("utf-8", "ignore"))
        t = t[:400000]
        p.write_text(t)
        return url, len(t)
    except Exception:
        p.write_text("")
        return url, 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = json.load(open(ROOT / "paper" / "sources.json"))
    urls = sorted({s["url"] for s in src})
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(fetch, urls))
    json.dump({u: key(u) for u in urls}, open(OUT / "index.json", "w"), indent=0)
    ok = sum(1 for _, n in res if n >= 2000)
    print(f"{len(urls)} urls, {ok} with >= 2000 characters of text")


if __name__ == "__main__":
    main()
