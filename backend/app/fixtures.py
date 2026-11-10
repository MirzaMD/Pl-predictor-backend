from __future__ import annotations

import html
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from footballdata_io import Client

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("FOOTBALL_IO_API_KEY")
PL_LEAGUE_ID = 15
CACHE_TTL = 3600          # 1 hour; keeps you far under the 1000-request limit
MAX_PAGES = 10            # safety cap on pagination
DEFAULT_WINDOW_DAYS = 7

_client = Client(API_KEY)
_cache: dict = {"ts": 0.0, "data": None}


def _market_probs(m: dict) -> dict | None:
    p = m.get("probabilities") or {}
    if not p:
        return None
    return {
        "H": p["home_win"] / 100,
        "D": p["draw"] / 100,
        "A": p["away_win"] / 100,
    }


def _unpack(body) -> tuple[list[dict], int]:
    """Return (matches, total_pages) for any of the likely response shapes."""
    if isinstance(body, list):
        return body, 1
    data = body.get("data", body)
    matches = data if isinstance(data, list) else data.get("matches", [])
    meta = body.get("meta") or {}
    total_pages = (meta.get("pagination") or {}).get("total_pages", 1)
    return matches, total_pages


def _fetch_all_incomplete() -> list[dict]:
    matches: list[dict] = []
    page = 1
    while page <= MAX_PAGES:
        body = _client.fixtures.today(league_id=PL_LEAGUE_ID, page=page, limit=50)
        batch, total_pages = _unpack(body)
        matches.extend(batch)
        if page >= total_pages:
            break
        page += 1
    return matches


def fetch_upcoming(days: int = DEFAULT_WINDOW_DAYS) -> list[dict]:
    now = time.time()
    if _cache["data"] is None or now - _cache["ts"] > CACHE_TTL:
        _cache["data"] = _fetch_all_incomplete()
        _cache["ts"] = now

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=days)
    fixtures = []
    for m in _cache["data"]:
        if m["league"]["league_id"] != PL_LEAGUE_ID or m["status"] != "incomplete":
            continue
        kickoff = datetime.strptime(m["match_date"], "%Y-%m-%d %H:%M:%S")
        if kickoff > cutoff:
            continue
        fixtures.append({
            "match_id": m["match_id"],
            "kickoff": m["match_date"],
            "game_week": m["game_week"],
            "home": html.unescape(m["home_team"]["team_name"]),
            "away": html.unescape(m["away_team"]["team_name"]),
            "home_logo": m["home_team"].get("team_logo"),   # new
            "away_logo": m["away_team"].get("team_logo"),   # new
            "market": _market_probs(m),
            "odds": m.get("odds"),
        })
    fixtures.sort(key=lambda f: f["kickoff"])
    return fixtures