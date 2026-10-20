import json, os, re, subprocess
from pathlib import Path
import yt_dlp
from dotenv import load_dotenv

load_dotenv()
JOBS_DIR = Path(__file__).parent / "jobs"
HOOK_WORDS = {"secret", "never", "mistake", "why", "how", "truth", "best", "worst", "nobody", "stop", "free", "money", "crazy", "biggest"}

def download(url, d):
    opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(d / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
    return d / "source.mp4", info.get("duration") or 0, info.get("title", "")

def load_audio(path):
    """Decode audio with FFmpeg into 16kHz mono float32 (bypasses PyAV version conflicts in faster-whisper)."""
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32)

def transcribe(path):
    """Transcribe audio with word-level timestamps and voice activity detection using faster-whisper."""
    from faster_whisper import WhisperModel
    model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), compute_type="int8")
    segs, _ = model.transcribe(load_audio(path), word_timestamps=True, vad_filter=True)
    segments, words = [], []
    for s in segs:
        segments.append((s.start, s.text.strip()))
        words += [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in (s.words or [])]
    return segments, words

def build_prompt(segments, duration, n):
    text = "\n".join(f"[{t:.0f}s] {s}" for t, s in segments)
    prompt = f"""You are an expert short-form video editor. Video length: {duration:.0f}s. Transcript with timestamps:
{text}

Pick the {n} best clips for YouTube Shorts. Rules: each 20-55 seconds; start exactly on a strong hook (a bold claim, question, surprise or story start); end on a complete thought; no overlaps; choose the most engaging, emotional, funny or valuable moments.
Return ONLY JSON: {{"clips":[{{"start":sec,"end":sec,"title":"catchy title under 8 words","score":0-100,"reason":"why this is a great hook"}}]}}"""
    return prompt

def parse_clips(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group())["clips"]

def pick_with_gemini(segments, duration, n):
    """Semantic hook selection via Google Gemini API."""
    import urllib.request
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    body = json.dumps({
        "contents": [{"parts": [{"text": build_prompt(segments, duration, n)}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        body,
        {"Content-Type": "application/json", "X-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        parts = json.load(r)["candidates"][0]["content"]["parts"]
    return parse_clips("".join(x.get("text", "") for x in parts))

def pick_with_claude(segments, duration, n):
    """Semantic hook selection via Anthropic Claude API."""
    import anthropic
    prompt = build_prompt(segments, duration, n)
    r = anthropic.Anthropic().messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_clips(r.content[0].text)

def pick(segments, words, duration, n):
    """Tiered AI hook selector with automatic failover to heuristic analyzer."""
    try:
        if os.getenv("GEMINI_API_KEY"):
            return pick_with_gemini(segments, duration, n)
        if os.getenv("ANTHROPIC_API_KEY"):
            return pick_with_claude(segments, duration, n)
    except Exception as e:
        print("AI hook picker failed, falling back to heuristic engine:", e)
    return pick_fallback(words, duration, n)

def pick_fallback(words, duration, n):
    """Algorithmic heuristic hook picker based on speech velocity, hook keywords, and punctuation cues."""
    cands = []
    for st in range(0, max(int(duration) - 40, 1), 10):
        ws = [w for w in words if st <= w["s"] < st + 40]
        score = len(ws) / 40 * 20 + sum(w["w"].lower().strip(".,!?") in HOOK_WORDS or "?" in w["w"] for w in ws) * 5
        cands.append((score, st))
    clips = []
    for score, st in sorted(cands, reverse=True):
        if all(abs(st - c["start"]) >= 45 for c in clips):
            clips.append({
                "start": st,
                "end": st + 40,
                "title": f"Best moment {len(clips)+1}",
                "score": min(int(score), 99),
                "reason": "High speech density + hook keywords detected",
            })
        if len(clips) == n:
            break
    return sorted(clips, key=lambda c: c["start"])

def snap(clips, words, duration):
    """Align clip temporal edges to real word acoustic boundaries to prevent mid-word cutting."""
    for c in clips:
        c["start"] = min((w["s"] for w in words if w["s"] >= c["start"] - 1), default=c["start"], key=lambda x: abs(x - c["start"]))
        c["end"] = min((w["e"] for w in words if w["e"] > c["start"]), default=c["end"], key=lambda x: abs(x - c["end"]))
        c["end"] = min(c["end"], duration or c["end"])
        c["start"], c["end"] = round(c["start"], 1), round(c["end"], 1)
    return clips

def ts(t):
    """Format seconds into ASS timestamp (H:MM:SS.cs)."""
    return f"{int(t//3600)}:{int(t%3600//60):02d}:{t%60:05.2f}"

def make_ass(words, start, end, title, path):
    """Generate kinetic ASS subtitles: 3-word chunks with bouncing highlighted word and intro hook banner."""
    head = """[Script Info]
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,Arial,92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,2,2,60,60,520,1
Style: Hook,Arial,74,&H0000E5FF,&H0000E5FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,8,70,70,230,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    clean = re.sub(r"[{}\\]", "", title)
    lines = [f"Dialogue: 1,{ts(0)},{ts(3)},Hook,,0,0,0,,{{\\fad(250,400)}}{clean}"]
    ws = [w for w in words if start <= w["s"] < end]
    for i in range(0, len(ws), 3):
        grp = ws[i:i+3]
        for j, w in enumerate(grp):
            s = w["s"] - start
            e = (grp[j+1]["s"] if j + 1 < len(grp) else w["e"]) - start
            txt = " ".join((f"{{\\c&H00E5FF&\\t(0,110,\\fscx118\\fscy118)\\t(110,200,\\fscx100\\fscy100)}}{x['w'].upper()}{{\\c&HFFFFFF&}}" if k == j else x["w"].upper()) for k, x in enumerate(grp))
            lines.append(f"Dialogue: 0,{ts(s)},{ts(max(e, s + .05))},Cap,,0,0,0,,{txt}")
    path.write_text(head + "\n".join(lines), encoding="utf-8")

def render(src, d, i, c, words):
    """Composites 9:16 vertical crop, slow dynamic retention zoom, and burnt-in vector kinetic subtitles."""
    dur = c["end"] - c["start"]
    ass = f"clip_{i}.ass"
    make_ass(words, c["start"], c["end"], c["title"], d / ass)
    vf = (
        f"crop=ih*9/16:ih,scale=1080:1920,scale=w='trunc(1080*(1+0.07*t/{dur:.2f})/2)*2':h=-2:eval=frame,"
        f"crop=1080:1920,ass={ass}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(c["start"]), "-t", f"{dur:.2f}",
            "-i", src.name, "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart", f"clip_{i}.mp4",
        ],
        cwd=d,
        check=True,
        capture_output=True,
    )

def process(url, job, upd, n=5):
    """End-to-end processing pipeline orchestrator."""
    d = JOBS_DIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    upd("Downloading video stream", 8)
    src, duration, title = download(url, d)
    job["video_title"] = title
    job["duration"] = duration

    upd("Transcribing audio (faster-whisper int8 + VAD)", 25)
    segments, words = transcribe(src)

    upd("Analyzing semantics & detecting viral hooks", 60)
    job["clips"] = snap(pick(segments, words, duration, n), words, duration)
    for k, c in enumerate(job["clips"], 1):
        c.update(id=k, file=None)

    for k, c in enumerate(job["clips"], 1):
        upd(f"Rendering clip {k} of {len(job['clips'])} (crop + kinetic captions + punch-in)", 65 + 33 * (k - 1) // len(job["clips"]))
        render(src, d, k, c, words)
        c["file"] = f"/files/{job['id']}/clip_{k}.mp4"
