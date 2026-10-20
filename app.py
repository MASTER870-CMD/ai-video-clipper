import threading, traceback, uuid, time, json, re, os, urllib.request
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Cookie
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pipeline
import youtube_api

pipeline.JOBS_DIR.mkdir(exist_ok=True)
app = FastAPI()
app.mount("/files", StaticFiles(directory=pipeline.JOBS_DIR), name="files")
JOBS = {}
SESSIONS = {}                       # session_id → {access_token, refresh_token, expires_at, user}


# ── Models ────────────────────────────────────────────────────────

class Req(BaseModel):
    url: str
    language: str = "auto"

class UploadReq(BaseModel):
    job_id: str
    clip_index: int
    title: str
    description: str = ""
    tags: List[str] = []
    privacy: str = "private"

class SeoReq(BaseModel):
    clip_title: str
    video_title: str = ""
    language: str = "en"


# ── Session helpers ───────────────────────────────────────────────

def _session(sid):
    if not sid or sid not in SESSIONS:
        return None
    return SESSIONS[sid]

def _token(session):
    """Return a valid access token, refreshing if expired."""
    if session.get("expires_at", 0) < time.time() and session.get("refresh_token"):
        try:
            t = youtube_api.refresh_access_token(session["refresh_token"])
            session["access_token"] = t["access_token"]
            session["expires_at"] = time.time() + t.get("expires_in", 3600)
        except Exception as e:
            print("Token refresh failed:", e)
            return None
    return session.get("access_token")


# ── Auth routes ───────────────────────────────────────────────────

@app.get("/auth/login")
def auth_login():
    """Redirect the user to the Google OAuth consent screen."""
    return RedirectResponse(youtube_api.get_auth_url(state=uuid.uuid4().hex))

@app.get("/auth/callback")
def auth_callback(code: str = "", error: str = ""):
    """Google redirects here after the user grants (or denies) permission."""
    if error:
        return RedirectResponse("/?error=" + error)
    if not code:
        return RedirectResponse("/?error=no_code")
    try:
        tokens = youtube_api.exchange_code(code)
        user = youtube_api.get_user_info(tokens["access_token"])
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token"),
            "expires_at": time.time() + tokens.get("expires_in", 3600),
            "user": user,
        }
        resp = RedirectResponse("/")
        resp.set_cookie("session_id", sid, httponly=True, max_age=86400 * 30)
        return resp
    except Exception:
        traceback.print_exc()
        return RedirectResponse("/?error=auth_failed")

@app.get("/auth/logout")
def auth_logout():
    resp = RedirectResponse("/")
    resp.delete_cookie("session_id")
    return resp


# ── API: current user ────────────────────────────────────────────

@app.get("/api/me")
def api_me(session_id: Optional[str] = Cookie(None)):
    s = _session(session_id)
    if not s:
        return {"logged_in": False}
    return {"logged_in": True, "user": s["user"]}


# ── API: YouTube videos with statistics ──────────────────────────

@app.get("/api/my-videos")
def api_my_videos(session_id: Optional[str] = Cookie(None)):
    s = _session(session_id)
    if not s:
        raise HTTPException(401, "Not logged in")
    tok = _token(s)
    if not tok:
        raise HTTPException(401, "Token expired — please log in again")
    try:
        search = youtube_api.list_my_videos(tok)
        items = search.get("items", [])
        if not items:
            return {"items": []}
        # Enrich with statistics (views, likes, comments)
        vids = [it["id"]["videoId"] for it in items if isinstance(it.get("id"), dict) and "videoId" in it["id"]]
        stats_map = {}
        if vids:
            details = youtube_api.get_video_details(tok, vids)
            for v in details.get("items", []):
                stats_map[v["id"]] = v.get("statistics", {})
        for it in items:
            vid = it["id"]["videoId"] if isinstance(it.get("id"), dict) else it.get("id")
            it["statistics"] = stats_map.get(vid, {})
        return {"items": items}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e)[:300])


# ── API: video comments ─────────────────────────────────────────

@app.get("/api/video-comments/{video_id}")
def api_comments(video_id: str, session_id: Optional[str] = Cookie(None)):
    s = _session(session_id)
    if not s:
        raise HTTPException(401, "Not logged in")
    tok = _token(s)
    if not tok:
        raise HTTPException(401, "Token expired")
    try:
        return youtube_api.get_comments(tok, video_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e)[:300])


# ── API: AI-generate SEO title, description, tags ────────────────

@app.post("/api/generate-seo")
def api_generate_seo(r: SeoReq):
    """Use Gemini to create a viral, SEO-optimized title, description, and tags."""
    prompt = f"""You are a top YouTube Shorts SEO expert. Generate viral metadata for a short-form clip.

Clip title: {r.clip_title}
Original video title: {r.video_title}
Language: {r.language}

Generate:
1. A viral, curiosity-driven title with emojis (max 80 chars). It should make people STOP scrolling.
2. An SEO description (2-3 punchy sentences + 5-8 trending hashtags). Include a call-to-action.
3. 10-12 relevant tags for discoverability.

Return ONLY JSON, no markdown:
{{"title": "...", "description": "...", "tags": ["tag1", "tag2", ...]}}"""

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        body,
        {"Content-Type": "application/json", "X-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            parts = json.load(resp)["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        data = json.loads(re.search(r"\{.*\}", text, re.S).group())
        return data
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"SEO generation failed: {str(e)[:200]}")


# ── API: upload clip to YouTube ──────────────────────────────────

@app.post("/api/upload-to-youtube")
def api_upload(r: UploadReq, session_id: Optional[str] = Cookie(None)):
    s = _session(session_id)
    if not s:
        raise HTTPException(401, "Not logged in")
    tok = _token(s)
    if not tok:
        raise HTTPException(401, "Token expired — please log in again")

    if r.job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    job = JOBS[r.job_id]
    clip = next((c for c in job.get("clips", []) if c["id"] == r.clip_index), None)
    if not clip:
        raise HTTPException(404, "Clip not found")

    path = pipeline.JOBS_DIR / r.job_id / f"clip_{r.clip_index}.mp4"
    if not path.exists():
        raise HTTPException(404, "Clip file not found on disk")

    try:
        result = youtube_api.upload_video(
            tok, str(path), r.title, r.description,
            tags=r.tags or None, privacy=r.privacy,
        )
        vid = result.get("id", "")
        return {"success": True, "video_id": vid, "url": f"https://youtube.com/watch?v={vid}"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e)[:300])


# ── Existing routes (clip creation) ──────────────────────────────

def _run(job):
    try:
        pipeline.process(job["url"], job, lambda s, p: job.update(stage=s, progress=p), language=job["language"])
        job.update(status="done", progress=100,
                   stage="Ready to post" if job["clips"] else job.get("note", "No clips found."))
    except Exception as e:
        traceback.print_exc()
        clean_err = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(e)).strip()
        job.update(status="error", error=clean_err[:300])

@app.post("/api/analyze")
def analyze(r: Req):
    target = r.url.strip().strip('"').strip("'")
    is_url = "youtu" in target or target.startswith("http://") or target.startswith("https://")
    is_file = os.path.exists(target) and os.path.isfile(target)
    if not is_url and not is_file:
        raise HTTPException(400, "Paste a YouTube link or a valid local video file path (.mp4)")
    lang = r.language.lower()
    if lang not in pipeline.SUPPORTED_LANGS:
        raise HTTPException(400, "Unsupported language")
    job = {"id": uuid.uuid4().hex[:10], "url": r.url, "language": lang, "status": "running",
           "stage": "Starting", "progress": 2, "clips": []}
    JOBS[job["id"]] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"id": job["id"]}

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[jid]

@app.get("/")
def index():
    return FileResponse("static/index.html")