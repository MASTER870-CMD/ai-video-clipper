# 🧪 AI Video Clipper — Prototype Workspace (`copy` branch)

> ⚠️ **Note for Reviewers & Recruiters**: This `copy` branch represents an experimental feature laboratory testing direct YouTube OAuth2 publishing, AI SEO metadata generation, and audio beat-sync detection. For the battle-tested, clean production engine, switch to the [`main`](https://github.com/MASTER870-CMD/ai-video-clipper/tree/main) branch.

---

## 🔬 Experimental Features in this Prototype

1. **Google OAuth2 YouTube Data API v3 Integration (`youtube_api.py`)**:
   - Web application OAuth flow for creator channel authorization (`/auth/login`, `/auth/callback`).
   - Token refresh handling with local session storage.
   - Resumable video upload protocol directly targeting YouTube Shorts channels.
2. **AI SEO & Tag Metadata Generator**:
   - Automatic generation of high-CTR Shorts titles, descriptions, and algorithmic tags using Gemini.
3. **Advanced Compositing Filters**:
   - Multi-scale punch-in zooms (`PUNCH_INS=1`).
   - Beat-sync audio markers with `librosa`.
   - Custom subtitle typography presets.

---

## 🛠 Prototype Setup

```bash
# 1. Install dependencies (including google-api and librosa)
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Set your GEMINI_API_KEY, GOOGLE_CLIENT_ID, and GOOGLE_CLIENT_SECRET in .env

# 3. Launch prototype server
uvicorn app:app --reload --port 8000
```
