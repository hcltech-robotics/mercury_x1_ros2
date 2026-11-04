"""Named places on a map ("kitchen", "entrance"), for the page and for voice ("take me to the kitchen").

Stored next to the map: ~/maps/<map>.places.json = [{"name", "x", "y", "yaw", "aliases": [...]}] in the map frame.
"""
from __future__ import annotations

import difflib
import json
import os
import re

ARTICLES = re.compile(r"^(the|a|an|our|my|your)\s+")


def norm(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    n = re.sub(r"\s+", " ", n).strip()
    n = ARTICLES.sub("", n)
    return re.sub(r"\s+(area|room|place|spot|point)$", "", n).strip()


def load(path: str) -> list[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return [p for p in data if isinstance(p, dict) and p.get("name")]
    except (OSError, ValueError):
        return []


def save(path: str, places: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(places, f, indent=1)
    os.replace(tmp, path)


def resolve(spoken: str, places: list[dict]) -> dict | None:
    """the place a spoken name means: exact / alias match after normalising, else the closest name (fuzzy)"""
    want = norm(spoken)
    if not want or not places:
        return None
    keys = {}
    for p in places:
        for n in [p["name"], *p.get("aliases", [])]:
            keys.setdefault(norm(n), p)
    if want in keys:
        return keys[want]
    for k, p in keys.items():   # "kitchen" vs "kitchen corner", "main entrance" vs "entrance"
        if k and (k in want.split() or want in k.split() or k in want or want in k):
            return p
    close = difflib.get_close_matches(want, list(keys), n=1, cutoff=0.75)
    return keys[close[0]] if close else None


def upsert(places: list[dict], name: str, x: float, y: float, yaw: float) -> list[dict]:
    key = norm(name)
    out = [p for p in places if norm(p["name"]) != key]
    old = next((p for p in places if norm(p["name"]) == key), {})
    out.append({"name": name.strip(), "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4),
                "aliases": old.get("aliases", [])})
    return out
