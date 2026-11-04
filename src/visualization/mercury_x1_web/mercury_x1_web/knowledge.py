"""Local knowledge for voice chat: markdown files -> small chunks -> BM25 keyword search (no model, a few ms).

Sources (all *.md, re-read when a file changes):
  <repo>/knowledge/               facts that ship with the code (HCLTech results, the lab, the robot)
  ~/.config/mercury_x1/knowledge/ local additions on the robot (same format)
Format: a "# title" line (shown to the model as the source, put the "as of" date there), "## section" headings,
paragraphs separated by blank lines. One fact per sentence works best; a paragraph is the unit of retrieval.
"""
from __future__ import annotations

import glob
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.realpath(__file__))
DIRS = [os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "knowledge")),
        os.path.expanduser("~/.config/mercury_x1/knowledge")]
MAX_CHUNK = 350   # characters; longer paragraphs are split at line ends
STOP = set("""a an the and or of to in on for at by with from as is are was were be been being it its this that these
those what which who whom whose when where why how do does did can could would should will shall may might must
i you he she we they me him her us them my your our their mercury tell please about there here than then so
not no yes if into over under up down out just also any some all many much work like know want have has had
get give say one s t whats""".replace("\n", " ").split())
SYNONYMS = {"hcl": "hcltech", "hcltechs": "hcltech", "employees": "headcount", "staff": "headcount",
            "people": "headcount", "employee": "headcount", "employed": "headcount", "boss": "ceo", "chief": "ceo", "profit": "income", "sales": "revenue",
            "turnover": "revenue", "earnings": "income", "workforce": "headcount"}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9₹$%]+(?:\.[0-9]+)?", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        if w in SYNONYMS:
            out.append(SYNONYMS[w])
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(SYNONYMS.get(w, w))
    return out


def chunks_of(path: str) -> list[dict]:
    title, section, out, para = os.path.basename(path), "", [], []

    def flush() -> None:
        text = "\n".join(para).strip()
        para.clear()
        if not text:
            return
        pieces, cur = [], ""
        for line in text.split("\n"):
            if cur and len(cur) + len(line) > MAX_CHUNK:
                pieces.append(cur)
                cur = ""
            cur = f"{cur}\n{line}" if cur else line
        pieces.append(cur)
        for p in pieces:
            out.append({"source": title, "section": section, "text": p.strip()})

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# "):
                flush()
                title = line[2:].strip()
            elif line.startswith("## ") or line.startswith("### "):
                flush()
                section = line.lstrip("#").strip()
            elif not line.strip():
                flush()
            else:
                para.append(line)
    flush()
    return out


class Knowledge:
    def __init__(self, dirs: list[str] | None = None) -> None:
        self.dirs = dirs or DIRS
        self._stamp: tuple = ()
        self.chunks: list[dict] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._avg = 1.0
        self.reload()

    def _files(self) -> list[str]:
        return sorted(p for d in self.dirs for p in glob.glob(os.path.join(d, "*.md"))
                      if os.path.basename(p).lower() != "readme.md")

    def reload(self) -> bool:
        files = self._files()
        stamp = tuple((p, os.path.getmtime(p)) for p in files)
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        self.chunks = [c for p in files for c in chunks_of(p)]
        self._tf = [Counter(tokens(f"{c['section']} {c['text']}")) for c in self.chunks]
        self._df = Counter(t for tf in self._tf for t in tf)
        self._avg = sum(sum(tf.values()) for tf in self._tf) / max(1, len(self._tf))
        return True

    def search(self, query: str, k: int = 6, rel: float = 0.4) -> list[dict]:
        """BM25 top k; drops hits scoring under `rel` x the best one (and everything when nothing matches well)"""
        self.reload()
        q = set(tokens(query))
        if not q or not self.chunks:
            return []
        n, k1, b = len(self.chunks), 1.5, 0.4   # mild length normalisation: client lists are long
        rare = {t for t in q if 0 < self._df[t] <= max(1, 0.3 * n)}   # words that actually tell chunks apart
        scored = []
        for i, tf in enumerate(self._tf):
            if not rare.intersection(tf):        # matching only common words is not a hit
                continue
            dl, s = sum(tf.values()), 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self._avg))
            scored.append((s, i))
        if not scored:
            return []
        scored.sort(reverse=True)
        best = scored[0][0]
        return [dict(self.chunks[i], score=round(s, 2)) for s, i in scored[:k] if s >= rel * best]


def _year(source: str) -> int:
    years = re.findall(r"20\d\d", source)
    return max(int(y) for y in years) if years else 0


def as_prompt(hits: list[dict]) -> str:
    if not hits:
        return ""
    hits = sorted(hits, key=lambda h: -_year(h["source"]))   # newest source first (stable: keeps rank within one)
    lines = [f"- [{h['source']}{' / ' + h['section'] if h['section'] else ''}] {h['text']}" for h in hits]
    return ("Facts for this question, newest source first. Use only what answers the question, copy names and numbers "
            "exactly, say the period of any number, and when two facts disagree trust the newer date:\n" + "\n".join(lines))
