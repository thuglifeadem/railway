"""
Auto Poster for X/Twitter — single‑file FastAPI app with scheduler

Features
- Web UI to set preferences: topics, sequence (random or round‑robin), cadence, time window, tone, hashtags, emojis, max posts/day, dry‑run.
- Background scheduler that posts automatically within your window.
- One‑click "Post now" and "Preview next 5".
- Optional AI copywriting via OpenAI if OPENAI_API_KEY is set, else a simple template generator is used.
- SQLite persistence.
- Tweepy for X/Twitter API v2 posting.

Quick start
1) pip install -U fastapi uvicorn jinja2 apscheduler tweepy python-dotenv pydantic
2) Set environment variables (see .env section below) or create a .env next to this file.
3) python auto_poster.py  then open http://127.0.0.1:8000

Environment (.env)
TW_BEARER_TOKEN=...
TW_CONSUMER_KEY=...
TW_CONSUMER_SECRET=...
TW_ACCESS_TOKEN=...
TW_ACCESS_TOKEN_SECRET=...
# Optional, for AI text generation
OPENAI_API_KEY=...

Notes
- You need a Twitter/X developer account with write permissions and user access tokens.
- For multiple accounts, duplicate the credentials with numeric suffixes (e.g., TW_ACCESS_TOKEN_2) and select in UI.
"""

import os
import random
import sqlite3
import string
from datetime import datetime, timedelta, time
from typing import List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
from dotenv import load_dotenv

# Twitter/X
try:
    import tweepy  # type: ignore
except Exception:  # pragma: no cover
    tweepy = None

# Optional OpenAI support
USE_OPENAI = False
try:
    from openai import OpenAI  # type: ignore
    USE_OPENAI = True
except Exception:
    USE_OPENAI = False

load_dotenv()

DB_PATH = os.environ.get("AUTO_POSTER_DB", "auto_poster.db")
app = FastAPI(title="Auto Poster for X/Twitter")

# ---- DB setup ----

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db()
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        account TEXT DEFAULT 'default',
        cadence_minutes INTEGER DEFAULT 240,
        start_hour INTEGER DEFAULT 8,
        end_hour INTEGER DEFAULT 22,
        sequence TEXT DEFAULT 'round',
        topics TEXT DEFAULT '',
        tone TEXT DEFAULT 'professional',
        hashtags TEXT DEFAULT '',
        use_emojis INTEGER DEFAULT 1,
        max_per_day INTEGER DEFAULT 4,
        dry_run INTEGER DEFAULT 1
    )
    """
)
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        last_post_at TEXT,
        posts_today INTEGER DEFAULT 0,
        topic_index INTEGER DEFAULT 0,
        posts_date TEXT
    )
    """
)
conn.commit()

# ensure rows exist
cur.execute("INSERT OR IGNORE INTO settings (id) VALUES (1)")
cur.execute("INSERT OR IGNORE INTO state (id, posts_date) VALUES (1, date('now'))")
conn.commit()

# ---- Models ----
class Settings(BaseModel):
    account: str = "default"
    cadence_minutes: int = 240
    start_hour: int = 8
    end_hour: int = 22
    sequence: str = "round"  # 'round' or 'random'
    topics: str = ""
    tone: str = "professional"  # 'professional', 'friendly', 'bold', 'playful'
    hashtags: str = ""
    use_emojis: int = 1
    max_per_day: int = 4
    dry_run: int = 1

# ---- Helpers ----

def get_settings() -> Settings:
    row = conn.execute("SELECT * FROM settings WHERE id=1").fetchone()
    return Settings(**dict(row))


def save_settings(s: Settings) -> None:
    conn.execute(
        """
        UPDATE settings SET account=?, cadence_minutes=?, start_hour=?, end_hour=?,
               sequence=?, topics=?, tone=?, hashtags=?, use_emojis=?, max_per_day=?, dry_run=?
        WHERE id=1
        """,
        (
            s.account,
            s.cadence_minutes,
            s.start_hour,
            s.end_hour,
            s.sequence,
            s.topics,
            s.tone,
            s.hashtags,
            s.use_emojis,
            s.max_per_day,
            s.dry_run,
        ),
    )
    conn.commit()


def get_state():
    return conn.execute("SELECT * FROM state WHERE id=1").fetchone()


def update_state(**kwargs):
    sets = ", ".join(f"{k}=?" for k in kwargs.keys())
    params = list(kwargs.values())
    params.append(1)
    conn.execute(f"UPDATE state SET {sets} WHERE id=?", params)
    conn.commit()


def within_window(now: datetime, start_h: int, end_h: int) -> bool:
    start = time(hour=start_h)
    end = time(hour=end_h)
    if start <= end:
        return start <= now.time() <= end
    # overnight window e.g. 22 -> 6
    return now.time() >= start or now.time() <= end


def tokenize_topics(raw: str) -> List[str]:
    t = [x.strip() for x in raw.split("\n") if x.strip()]
    return t

EMOJI_BANK = {
    "professional": ["📈", "🧠", "🛠️"],
    "friendly": ["😊", "✨", "🙌"],
    "bold": ["🔥", "🚀", "⚡"],
    "playful": ["😎", "🎯", "🤖"],
}


def simple_copy(topic: str, tone: str, hashtags: str, use_emojis: bool) -> str:
    openers = {
        "professional": [
            "Key insight:",
            "Quick take:",
            "Today’s thought:",
        ],
        "friendly": [
            "Little idea:",
            "Hey builders:",
            "Sharing this:",
        ],
        "bold": [
            "Unpopular opinion:",
            "Hot take:",
            "Real talk:",
        ],
        "playful": [
            "Fun fact:",
            "Tiny hack:",
            "Here’s a nugget:",
        ],
    }
    opener = random.choice(openers.get(tone, openers["professional"]))
    body = f"{opener} {topic}."
    if use_emojis:
        body += " " + random.choice(EMOJI_BANK.get(tone, EMOJI_BANK["professional"]))
    if hashtags.strip():
        tags = " ".join("#" + h.strip().lstrip("#") for h in hashtags.split(",") if h.strip())
        if tags:
            body += " " + tags
    # ensure <= 280
    return body[:279]


def ai_copy(topic: str, tone: str, hashtags: str, use_emojis: bool) -> str:
    if not USE_OPENAI or not os.environ.get("OPENAI_API_KEY"):
        return simple_copy(topic, tone, hashtags, use_emojis)
    try:
        client = OpenAI()
        style = {
            "professional": "concise, data‑driven, neutral",
            "friendly": "warm, encouraging, simple",
            "bold": "strong, decisive, energetic",
            "playful": "light, witty, fun",
        }.get(tone, "concise, neutral")
        emoji_hint = "Include one fitting emoji." if use_emojis else "Do not include emojis."
        tag_hint = (
            f"Append these hashtags if natural: {hashtags}."
            if hashtags.strip()
            else "Do not add hashtags."
        )
        prompt = (
            "Write one tweet under 240 characters. Topic: '" + topic + "'. "
            + f"Tone: {style}. {emoji_hint} {tag_hint} No quotes, no hashtags repetition."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You craft crisp, original tweets."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=120,
        )
        text = resp.choices[0].message.content.strip()
        # safety clamp
        return text[:279]
    except Exception:
        return simple_copy(topic, tone, hashtags, use_emojis)


# ---- Twitter client ----

def get_twitter_client(account_label: str = "default"):
    if tweepy is None:
        return None

    suffix = "" if account_label == "default" else f"_{account_label}"

    bearer = os.getenv(f"TW_BEARER_TOKEN{suffix}") or os.getenv("TW_BEARER_TOKEN")
    ck = os.getenv(f"TW_CONSUMER_KEY{suffix}") or os.getenv("TW_CONSUMER_KEY")
    cs = os.getenv(f"TW_CONSUMER_SECRET{suffix}") or os.getenv("TW_CONSUMER_SECRET")
    at = os.getenv(f"TW_ACCESS_TOKEN{suffix}") or os.getenv("TW_ACCESS_TOKEN")
    ats = os.getenv(f"TW_ACCESS_TOKEN_SECRET{suffix}") or os.getenv("TW_ACCESS_TOKEN_SECRET")

    if not all([bearer, ck, cs, at, ats]):
        return None

    client = tweepy.Client(
        bearer_token=bearer,
        consumer_key=ck,
        consumer_secret=cs,
        access_token=at,
        access_token_secret=ats,
    )
    return client


# ---- Posting logic ----

def maybe_reset_counter(now: datetime):
    st = get_state()
    today_str = now.strftime("%Y-%m-%d")
    if st["posts_date"] != today_str:
        update_state(posts_today=0, posts_date=today_str)


def pick_topic(settings: Settings) -> Optional[str]:
    topics = tokenize_topics(settings.topics)
    if not topics:
        return None
    st = get_state()
    if settings.sequence == "random":
        return random.choice(topics)
    # round robin
    idx = st["topic_index"] or 0
    topic = topics[idx % len(topics)]
    update_state(topic_index=(idx + 1) % len(topics))
    return topic


def perform_post(text: str, settings: Settings) -> dict:
    if settings.dry_run:
        return {"status": "dry_run", "text": text}

    client = get_twitter_client(settings.account)
    if client is None:
        return {"status": "error", "error": "Twitter client not configured or tweepy missing."}

    try:
        resp = client.create_tweet(text=text)
        return {"status": "ok", "tweet_id": str(resp.data.get("id")), "text": text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def should_post(now: datetime, settings: Settings) -> bool:
    st = get_state()
    maybe_reset_counter(now)

    # within time window
    if not within_window(now, settings.start_hour, settings.end_hour):
        return False

    # cadence gate
    last = st["last_post_at"]
    if last:
        last_dt = datetime.fromisoformat(last)
        if now < last_dt + timedelta(minutes=settings.cadence_minutes):
            return False

    # daily cap
    if (st["posts_today"] or 0) >= settings.max_per_day:
        return False

    return True


def generate_post(settings: Settings) -> Optional[str]:
    topic = pick_topic(settings)
    if not topic:
        return None
    return ai_copy(topic, settings.tone, settings.hashtags, bool(settings.use_emojis))


# ---- Scheduler ----

scheduler = BackgroundScheduler()


def tick():
    now = datetime.now()
    s = get_settings()
    if not should_post(now, s):
        return

    text = generate_post(s)
    if not text:
        return

    result = perform_post(text, s)
    if result.get("status") == "ok":
        update_state(last_post_at=now.isoformat(), posts_today=(get_state()["posts_today"] or 0) + 1)


scheduler.add_job(tick, IntervalTrigger(minutes=1), id="poster_tick", replace_existing=True)
try:
    scheduler.start()
except Exception:
    pass

# ---- UI ----

HTML_BASE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auto Poster for X/Twitter</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 2rem; }
    .wrap { max-width: 980px; margin: 0 auto; }
    h1 { margin: 0 0 1rem 0; }
    form { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    textarea { width: 100%; min-height: 160px; }
    .full { grid-column: 1 / -1; }
    .row { display: flex; gap: .75rem; align-items: center; }
    input[type="number"] { width: 120px; }
    .card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 1rem; margin: 1rem 0; }
    .actions { display: flex; gap: .5rem; }
    button { padding: .6rem 1rem; border-radius: 12px; border: 1px solid #d1d5db; background: white; cursor: pointer; }
    button.primary { background: #111827; color: white; border-color: #111827; }
    .hint { color: #6b7280; font-size: 0.9rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .badge { padding: .2rem .5rem; border: 1px solid #e5e7eb; border-radius: 999px; font-size: .8rem; }
    .ok { color: #065f46; }
    .err { color: #7f1d1d; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Auto Poster for X/Twitter</h1>

    <div class="card">
      <div class="row">
        <form method="post" action="/save">
          <div class="row"><label>Account label</label>
            <input type="text" name="account" value="{account}" />
            <span class="hint">Use suffix to pick env vars like <span class="mono">TW_ACCESS_TOKEN_mylabel</span></span>
          </div>
          <div class="row"><label>Cadence minutes</label><input type="number" name="cadence_minutes" value="{cadence}" min="5" max="1440" /></div>
          <div class="row"><label>Start hour</label><input type="number" name="start_hour" value="{start_h}" min="0" max="23" /></div>
          <div class="row"><label>End hour</label><input type="number" name="end_hour" value="{end_h}" min="0" max="23" /></div>
          <div class="row"><label>Sequence</label>
            <select name="sequence">
              <option value="round" {round_sel}>Round‑robin</option>
              <option value="random" {rand_sel}>Random</option>
            </select>
          </div>
          <div class="row"><label>Tone</label>
            <select name="tone">
              <option value="professional" {t_prof}>Professional</option>
              <option value="friendly" {t_fri}>Friendly</option>
              <option value="bold" {t_bold}>Bold</option>
              <option value="playful" {t_play}>Playful</option>
            </select>
          </div>
          <div class="row"><label>Use emojis</label>
            <select name="use_emojis">
              <option value="1" {e_on}>On</option>
              <option value="0" {e_off}>Off</option>
            </select>
          </div>
          <div class="row"><label>Max posts per day</label><input type="number" name="max_per_day" value="{maxpd}" min="1" max="24" /></div>
          <div class="row"><label>Dry‑run mode</label>
            <select name="dry_run">
              <option value="1" {d_on}>On</option>
              <option value="0" {d_off}>Off</option>
            </select>
          </div>
          <div class="full">
            <label>Topics, one per line</label>
            <textarea name="topics" placeholder="e.g. Data governance tip…\nAzure cost optimization…\nAI + energy sector insight…">{topics}</textarea>
          </div>
          <div class="full">
            <label>Hashtags (comma separated, optional)</label>
            <input type="text" name="hashtags" value="{hashtags}" />
          </div>
          <div class="actions full">
            <button class="primary" type="submit">Save</button>
            <a href="/preview"><button type="button">Preview next 5</button></a>
            <form method="post" action="/post-now" style="display:inline"><button type="submit">Post now</button></form>
            <a href="/health"><button type="button">Health</button></a>
          </div>
        </form>
      </div>
      <div class="hint">Status: {status_html}</div>
    </div>

    <div class="card">
      <div>Env check: {env_status}</div>
      <div class="hint">Set <span class="mono">TW_*\nOPENAI_API_KEY</span> in environment or .env</div>
    </div>
  </div>
</body>
</html>
"""


def env_ok(account: str) -> bool:
    client = get_twitter_client(account)
    return client is not None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    s = get_settings()
    st = get_state()
    status_bits = []
    if st["last_post_at"]:
        status_bits.append(f"Last post at {st['last_post_at']}")
    status_bits.append(f"Posts today: {st['posts_today']}")
    status_bits.append(f"Next eligible in ~{s.cadence_minutes} min after last post")
    status_html = " | ".join(status_bits)

    html = HTML_BASE.format(
        account=s.account,
        cadence=s.cadence_minutes,
        start_h=s.start_hour,
        end_h=s.end_hour,
        round_sel="selected" if s.sequence == "round" else "",
        rand_sel="selected" if s.sequence == "random" else "",
        t_prof="selected" if s.tone == "professional" else "",
        t_fri="selected" if s.tone == "friendly" else "",
        t_bold="selected" if s.tone == "bold" else "",
        t_play="selected" if s.tone == "playful" else "",
        e_on="selected" if s.use_emojis else "",
        e_off="selected" if not s.use_emojis else "",
        maxpd=s.max_per_day,
        d_on="selected" if s.dry_run else "",
        d_off="selected" if not s.dry_run else "",
        topics=s.topics,
        hashtags=s.hashtags,
        status_html=status_html,
        env_status=(
            "<span class='badge ok'>Twitter client configured</span>"
            if env_ok(s.account)
            else "<span class='badge err'>Twitter client NOT configured</span>"
        ),
    )
    return HTMLResponse(html)


@app.post("/save")
async def save(
    account: str = Form(...),
    cadence_minutes: int = Form(...),
    start_hour: int = Form(...),
    end_hour: int = Form(...),
    sequence: str = Form(...),
    tone: str = Form(...),
    use_emojis: int = Form(...),
    max_per_day: int = Form(...),
    dry_run: int = Form(...),
    topics: str = Form(""),
    hashtags: str = Form(""),
):
    s = Settings(
        account=account.strip() or "default",
        cadence_minutes=max(5, min(1440, cadence_minutes)),
        start_hour=max(0, min(23, start_hour)),
        end_hour=max(0, min(23, end_hour)),
        sequence=sequence if sequence in ("round", "random") else "round",
        topics=topics.strip(),
        tone=tone if tone in ("professional", "friendly", "bold", "playful") else "professional",
        hashtags=hashtags.strip(),
        use_emojis=1 if use_emojis else 0,
        max_per_day=max(1, min(24, max_per_day)),
        dry_run=1 if dry_run else 0,
    )
    save_settings(s)
    return RedirectResponse("/", status_code=303)


@app.get("/preview")
async def preview():
    s = get_settings()
    topics = tokenize_topics(s.topics)
    if not topics:
        return JSONResponse({"error": "No topics configured yet."}, status_code=400)

    out = []
    for _ in range(min(5, len(topics))):
        topic = random.choice(topics) if s.sequence == "random" else topics[_ % len(topics)]
        out.append(ai_copy(topic, s.tone, s.hashtags, bool(s.use_emojis)))
    return JSONResponse({"preview": out, "dry_run": bool(s.dry_run)})


@app.post("/post-now")
async def post_now():
    s = get_settings()
    text = generate_post(s)
    if not text:
        return JSONResponse({"error": "Add at least one topic first."}, status_code=400)
    res = perform_post(text, s)
    if res.get("status") == "ok":
        now = datetime.now()
        st = get_state()
        update_state(last_post_at=now.isoformat(), posts_today=(st["posts_today"] or 0) + 1)
    return JSONResponse(res)


@app.get("/health")
async def health():
    s = get_settings()
    st = get_state()
    return {
        "scheduler": "running",
        "env_twitter_configured": env_ok(s.account),
        "settings": s.model_dump(),
        "state": dict(st),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
