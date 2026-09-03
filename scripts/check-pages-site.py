#!/usr/bin/env python3
"""Verify static showcase files, local links, and honest-copy constraints."""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1] / "docs"
PAGES = [ROOT / "index.html", ROOT / "README.html", ROOT / "404.html"]

INVENTED_PATTERNS = [
    r"arXiv",
    r"accepted at",
    r"published in",
    r"\bSOTA\b",
    r"61\.91",
    r"63\.45",
    r"63\.11",
    r"64\.58",
]


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.ids: set[str] = set()
        self.lang_ok = False
        self.title = ""
        self.has_viewport = False
        self.has_description = False
        self.has_canonical = False
        self.in_title = False
        self.skip_link = False
        self.h1 = 0
        self.main = 0
        self.alts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: v or "" for k, v in attrs}
        if tag == "html" and ad.get("lang", "").lower().startswith("zh"):
            self.lang_ok = True
        if tag == "title":
            self.in_title = True
        if tag == "meta" and ad.get("name") == "viewport":
            self.has_viewport = True
        if tag == "meta" and ad.get("name") == "description" and ad.get("content"):
            self.has_description = True
        if tag == "link" and ad.get("rel") == "canonical" and ad.get("href"):
            self.has_canonical = True
        if tag == "a":
            href = ad.get("href")
            if href:
                self.hrefs.append(href)
            if "skip-link" in ad.get("class", "") or href == "#main":
                self.skip_link = True
        if tag in {"img", "script", "link"} and ad.get("href"):
            self.srcs.append(ad["href"])
        if tag in {"img", "script"} and ad.get("src"):
            self.srcs.append(ad["src"])
        if tag == "img":
            self.alts.append(ad.get("alt", ""))
        if "id" in ad:
            self.ids.add(ad["id"])
        if tag == "h1":
            self.h1 += 1
        if tag == "main":
            self.main += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data


def is_local(url: str) -> bool:
    if url.startswith(("http://", "https://", "mailto:", "data:")):
        return False
    return True


def resolve(page: Path, url: str) -> Path | None:
    raw = unquote(url.split("?", 1)[0])
    if raw.startswith("#"):
        return None
    path = raw.split("#", 1)[0]
    if not path:
        return None
    return (page.parent / path).resolve()


def main() -> int:
    errors: list[str] = []
    for page in PAGES:
        if not page.exists():
            errors.append(f"missing {page.name}")
            continue
        text = page.read_text(encoding="utf-8")
        parser = LinkParser()
        parser.feed(text)
        if page.name != "404.html":
            if not parser.lang_ok:
                errors.append(f"{page.name}: html lang should be zh")
            if not parser.title:
                errors.append(f"{page.name}: missing title")
            if not parser.has_viewport:
                errors.append(f"{page.name}: missing viewport")
            if not parser.has_description:
                errors.append(f"{page.name}: missing description")
            if not parser.has_canonical:
                errors.append(f"{page.name}: missing canonical")
        if parser.h1 != 1:
            errors.append(f"{page.name}: expected one h1, got {parser.h1}")
        if parser.main != 1:
            errors.append(f"{page.name}: expected one main, got {parser.main}")
        if not parser.skip_link:
            errors.append(f"{page.name}: missing skip link")
        for alt in parser.alts:
            if not alt.strip():
                errors.append(f"{page.name}: image missing alt")
        for href in parser.hrefs:
            if href.startswith("#"):
                frag = href[1:]
                if frag and frag not in parser.ids:
                    errors.append(f"{page.name}: broken fragment #{frag}")
                continue
            if is_local(href):
                target = resolve(page, href)
                if target is None:
                    continue
                if not target.exists():
                    errors.append(f"{page.name}: broken local link {href}")
        for src in parser.srcs:
            if is_local(src):
                target = resolve(page, src)
                if target is not None and not target.exists():
                    errors.append(f"{page.name}: missing asset {src}")
        for pat in INVENTED_PATTERNS:
            if re.search(pat, text, flags=re.I):
                errors.append(f"{page.name}: forbidden invented-claim pattern {pat}")

    required_assets = [
        ROOT / "assets" / "ChatRWKV.png",
        ROOT / "assets" / "RWKV-eval.png",
        ROOT / "assets" / "styles.css",
        ROOT / "assets" / "favicon.svg",
        ROOT / "robots.txt",
        ROOT / "sitemap.xml",
        ROOT / ".nojekyll",
    ]
    for asset in required_assets:
        if not asset.exists():
            errors.append(f"missing {asset.relative_to(ROOT)}")

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("OK: static files, local links, a11y landmarks, and honesty checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
