"""YouTube OAuth 2.0 + Data API v3 helper.

Handles:
- Google OAuth consent URL generation
- Auth code → access/refresh token exchange
- Token refresh
- User profile lookup
- Listing user's YouTube videos (with statistics)
- Fetching video comments
- Resumable video upload to YouTube
"""

import json, os, urllib.request, urllib.parse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

SCOPES = " ".join([
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
])


# ── OAuth helpers ──────────────────────────────────────────────────

def get_auth_url(state=""):
    """Build the Google OAuth consent URL."""
    params = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    })
    return "https://accounts.google.com/o/oauth2/auth?" + params


def exchange_code(code):
    """Exchange an authorization code for access + refresh tokens."""
    data = urllib.parse.urlencode({
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token", data,
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def refresh_access_token(refresh_token):
    """Use a refresh token to get a fresh access token."""
    data = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token", data,
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ── User info ──────────────────────────────────────────────────────

def get_user_info(access_token):
    """Fetch Google profile (name, picture, id)."""
    req = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


# ── YouTube Data API v3 ───────────────────────────────────────────

def list_my_videos(access_token, max_results=20):
    """List the authenticated user's most recent uploads (IDs only from search)."""
    params = urllib.parse.urlencode({
        "part": "snippet",
        "forMine": "true",
        "type": "video",
        "order": "date",
        "maxResults": str(max_results),
    })
    req = urllib.request.Request(
        f"https://www.googleapis.com/youtube/v3/search?{params}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def get_video_details(access_token, video_ids):
    """Get snippet + statistics for a list of video IDs."""
    if not video_ids:
        return {"items": []}
    ids = ",".join(video_ids[:50])
    params = urllib.parse.urlencode({
        "part": "snippet,statistics",
        "id": ids,
    })
    req = urllib.request.Request(
        f"https://www.googleapis.com/youtube/v3/videos?{params}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def get_comments(access_token, video_id, max_results=25):
    """Fetch top-level comment threads for a video."""
    params = urllib.parse.urlencode({
        "part": "snippet",
        "videoId": video_id,
        "order": "relevance",
        "maxResults": str(max_results),
        "textFormat": "plainText",
    })
    req = urllib.request.Request(
        f"https://www.googleapis.com/youtube/v3/commentThreads?{params}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def upload_video(access_token, file_path, title, description="", tags=None, privacy="private"):
    """Upload a video via the YouTube resumable-upload API.

    Steps:
    1. POST metadata to initiate a resumable upload → get an upload URI.
    2. PUT the entire video file to that URI.
    Returns the YouTube API response (contains the new video ID).
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size

    metadata = json.dumps({
        "snippet": {
            "title": title[:100],
            "description": description or "Created with Cliping",
            "tags": tags or ["shorts", "clip"],
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }).encode("utf-8")

    init = urllib.request.Request(
        "https://www.googleapis.com/upload/youtube/v3/videos"
        "?uploadType=resumable&part=snippet,status",
        data=metadata,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(file_size),
        },
        method="POST",
    )
    with urllib.request.urlopen(init, timeout=60) as r:
        upload_url = r.headers["Location"]

    with open(file_path, "rb") as f:
        video_data = f.read()

    put = urllib.request.Request(
        upload_url,
        data=video_data,
        headers={
            "Content-Type": "video/mp4",
            "Content-Length": str(file_size),
        },
        method="PUT",
    )
    with urllib.request.urlopen(put, timeout=300) as r:
        return json.load(r)
