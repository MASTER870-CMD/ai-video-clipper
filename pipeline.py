import bisect, json, os, re, subprocess, time
from pathlib import Path
import numpy as np
import yt_dlp
from dotenv import load_dotenv

load_dotenv()
JOBS_DIR = Path(__file__).parent / "jobs"
HOOK_WORDS = {"secret", "never", "mistake", "why", "how", "truth", "best", "worst", "nobody", "stop",
              "free", "money", "crazy", "biggest", "shocking", "warning", "insane", "unbelievable"}
CLIP_COUNT = int(os.getenv("CLIP_COUNT", "6"))       # clips to make per video
MIN_FINAL = int(os.getenv("MIN_FINAL_SCORE", "65"))  # 0-100, at or above this a clip counts as a strong pick
PUNCH_INS = os.getenv("PUNCH_INS", "1") == "1"       # quick zoom pop every 2 seconds
CAPTIONS = os.getenv("CAPTIONS", "1") == "1"         # word captions on speech
RES = 2                                              # picture and sound samples per second
MIN_LEN, MAX_LEN = 15, 60                            # clip length limits in seconds
GAP = 3.0                                            # seconds of space between clips
END_PUNCT = (".", "?", "!", "।", "…")
LANG_CODES = {"english": "en", "tamil": "ta", "telugu": "te", "hindi": "hi", "malayalam": "ml", "kannada": "kn"}
LANG_NAMES = {"en": "English", "ta": "Tamil", "te": "Telugu", "hi": "Hindi", "ml": "Malayalam", "kn": "Kannada"}
INDIC = {"ta", "te", "hi", "ml", "kn"}
SUPPORTED_LANGS = {"auto", *LANG_CODES}
CAPTION_FONT = os.getenv("CAPTION_FONT", "Nirmala UI")  # ships with Windows, covers all five Indian scripts
HYPE_TITLES = ["Wait for it", "Nobody saw this coming", "This part is insane",
               "Watch till the end", "You need to hear this", "Don't skip this"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def is_file_locked(e):
    """Windows error 32: another program (antivirus, search indexer) is holding the file."""
    return getattr(e, "winerror", None) == 32 or "WinError 32" in str(e) or "used by another process" in str(e)


def download(url, d):
    # Support local video files directly (bypasses YouTube entirely)
    local_p = Path(url.strip('"').strip("'"))
    if local_p.exists() and local_p.is_file():
        dest = d / "source.mp4"
        if local_p.resolve() != dest.resolve():
            import shutil
            shutil.copy2(local_p, dest)
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(dest)],
                capture_output=True, text=True, check=True
            )
            duration = float(probe.stdout.strip() or 0)
        except Exception:
            duration = 0
        return dest, duration, local_p.stem, ""

    cookie_file = Path("cookies.txt")
    opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(d / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "keepvideo": True,
        "retries": 10,
        "fragment_retries": 10,
    }
    if cookie_file.exists():
        opts["cookiefile"] = str(cookie_file.resolve())

    info = None
    for attempt in range(1, 9):
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                info = y.extract_info(url, download=True)
            break
        except Exception as e:
            if attempt == 8 or not is_file_locked(e):
                err_msg = str(e)
                if "Sign in to confirm you’re not a bot" in err_msg or "Sign in to confirm you're not a bot" in err_msg:
                    raise RuntimeError("YouTube bot-check triggered for this video. Export cookies to cookies.txt or try another video.") from e
                raise
            print(f"File locked by Windows, retrying in {attempt * 3}s:", e)
            time.sleep(attempt * 3)

    for f in d.glob("source.*"):  # drop the separate video/audio parts after merging
        if f.name != "source.mp4":
            try:
                f.unlink()
            except OSError:
                pass
    uploader = (info or {}).get("uploader") or (info or {}).get("channel") or ""
    return d / "source.mp4", (info or {}).get("duration") or 0, (info or {}).get("title", ""), uploader


def load_audio(path):
    """Decode audio with FFmpeg (avoids the PyAV version bug in faster-whisper)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def fit(arr, n):
    out = np.zeros(n, np.float32)
    k = min(n, len(arr))
    out[:k] = arr[:k]
    return out


def visual_profile(src, n):
    """Per half-second: how much the picture changes (motion) and how bright it is."""
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-an", "-vf",
                              f"fps={RES},scale=64:36,format=gray", "-f", "rawvideo", "-"],
                             capture_output=True, check=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, 36, 64).astype(np.float32)
    except Exception as e:
        print("Picture scan skipped:", e)
        fr = np.zeros((0, 36, 64), np.float32)
    if len(fr) < 2:
        return np.zeros(n, np.float32), np.zeros(n, np.float32)
    motion = np.zeros(len(fr), np.float32)
    motion[1:] = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    return fit(motion, n), fit(fr.mean(axis=(1, 2)), n)


def audio_energy(audio, n):
    """Loudness per half-second."""
    hop = 16000 // RES
    k = len(audio) // hop
    if k == 0:
        return np.zeros(n, np.float32)
    rms = np.sqrt((audio[:k * hop].reshape(k, hop) ** 2).mean(axis=1))
    return fit(rms, n)


def pct(arr):
    """Rank each half-second against the whole video: 0 = calmest, 1 = most intense."""
    if len(arr) < 2 or float(arr.max()) == float(arr.min()):
        return np.full(len(arr), 0.5, np.float32)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), np.float32)
    ranks[order] = np.linspace(0, 1, len(arr), dtype=np.float32)
    return ranks


def build_signals(src, audio, duration):
    n = max(1, int(duration * RES))
    motion, bright = visual_profile(src, n)
    energy = audio_energy(audio, n)
    return {"act": (pct(motion) + pct(energy)) / 2, "bright": pct(bright)}


def window_score(s, e, sig):
    """0-100: how eye-catching a stretch is. The first 3 seconds count most, because that's what stops the scroll."""
    total = len(sig["act"])
    i0 = min(int(s * RES), total - 1)
    i1 = max(min(int(e * RES), total), i0 + 1)
    hook = float(sig["act"][i0:min(i0 + 3 * RES, i1)].mean())
    body = float(sig["act"][i0:i1].mean())
    light = float(sig["bright"][i0:i1].mean())
    return 100 * (0.45 * hook + 0.35 * body + 0.20 * light)


def signal_windows(duration, sig):
    """Many overlapping stretches, scored by picture and sound. The picker chooses among them."""
    out = []
    for s in np.arange(0, max(duration - MIN_LEN, 0), 1.5):
        s = float(s)
        for length in (25, 35, 45):
            e = min(s + length, duration)
            if e - s >= MIN_LEN:
                out.append({"start": s, "end": e, "final": window_score(s, e, sig),
                            "title": "", "reason": "High motion, loudness and brightness"})
    return out


def window_title(words, s, e):
    said = [w["w"] for w in words if s <= w["s"] < e][:6]
    return (" ".join(said))[:60]


def detect_beats(audio):
    """Beat times in seconds, so captions can pop on the music. Empty list if librosa is missing or disabled."""
    if os.getenv("BEAT_SYNC", "1") != "1":
        return []
    try:
        import librosa
        _, frames = librosa.beat.beat_track(y=audio, sr=16000)
        return sorted(librosa.frames_to_time(frames, sr=16000).tolist())
    except Exception as e:
        print("Beat detection skipped:", e)
        return []


_model = None
def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        threads = min(8, os.cpu_count() or 4)
        _model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), device=os.getenv("WHISPER_DEVICE", "cpu"),
                              compute_type=os.getenv("WHISPER_COMPUTE", "int8"),
                              cpu_threads=threads)
    return _model


def transcribe(audio, language="auto", cache_path=None):
    """Transcribe audio with optional on‑disk caching.
    If *cache_path* points to a JSON file and exists, the cached result is returned.
    The cache stores the tuple (segments, words, language).
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["segments"], cached["words"], cached["language"]
        except Exception:
            pass  # fall back to fresh transcription
    lang = LANG_CODES.get(language, language)
    lang = None if lang == "auto" else lang
    segs, info = get_model().transcribe(audio, language=lang, word_timestamps=True, vad_filter=True, beam_size=1)
    segments, words = [], []
    for s in segs:
        segments.append((s.start, s.text.strip()))
        words += [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in (s.words or []) if w.word.strip()]
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"segments": segments, "words": words, "language": info.language}, f)
        except Exception:
            pass
    return segments, words, info.language


def build_prompt(segments, duration, n, lang):
    text = "\n".join(f"[{t:.0f}s] {s}" for t, s in segments) or "(no speech detected: music or silent footage)"
    return f"""You are a viral short-form editor for Instagram Reels and YouTube Shorts. The speech is in {LANG_NAMES.get(lang, 'English')}. Video length: {duration:.0f}s.

Transcript with timestamps:
{text}

Find {n} candidate clips that are most likely to go viral. Spread them across different parts of the video, and list the strongest first.
What makes a clip go viral:
- The first 3 seconds grab attention: a bold or shocking claim, a sharp question, a specific number, a conflict, a confession, a joke, or a "here's why" promise.
- The payoff is surprising, funny, emotional, or impressive, and it lands before the viewer scrolls away.
- It is one complete idea that ends cleanly.
Rules:
- Each clip is 20-45 seconds and starts 0-1 seconds before its hook.
- Never start on a greeting, intro, sponsor read, filler, recap, or anything that needs earlier context.
- No overlapping clips.
- If the transcript is thin (music or visuals), choose the most energetic stretch and score it honestly.
- Titles: curiosity-driving, in the same language as the speech, under 8 words, never misleading.

Return ONLY JSON, no markdown:
{{"clips":[{{"start":seconds,"end":seconds,"hook_line":"...","hook_score":0-100,"score":0-100,"title":"...","reason":"one short sentence"}}]}}
hook_score = how strongly the first 3 seconds grab attention. score = overall viral potential."""


def parse_clips(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group())["clips"]


def pick_with_gemini(segments, duration, n, lang):
    import urllib.request
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    body = json.dumps({"contents": [{"parts": [{"text": build_prompt(segments, duration, n, lang)}]}],
                       "generationConfig": {"responseMimeType": "application/json"}}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", body,
                                 {"Content-Type": "application/json", "X-goog-api-key": os.environ["GEMINI_API_KEY"]})
    with urllib.request.urlopen(req, timeout=120) as r:
        parts = json.load(r)["candidates"][0]["content"]["parts"]
    return parse_clips("".join(x.get("text", "") for x in parts))


def pick_with_claude(segments, duration, n, lang):
    import anthropic
    r = anthropic.Anthropic().messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=2000,
                                              messages=[{"role": "user",
                                                         "content": build_prompt(segments, duration, n, lang)}])
    return parse_clips(r.content[0].text)


def snap_one(c, starts, ends, words, duration):
    """Start on a real word and end on a sentence-ending word, so clips never cut mid-word or mid-thought."""
    s = c["start"]
    i = bisect.bisect_left(starts, s - 0.3)
    if i < len(starts) and starts[i] <= s + 1.5:
        s = starts[i]
    s = max(0.0, s - 0.15)
    e = c["end"]
    lo, hi = bisect.bisect_left(ends, e - 4), bisect.bisect_right(ends, e + 3)
    near = words[lo:hi]
    if near:
        sentence = [w for w in near if w["w"].endswith(END_PUNCT)] or near
        e = min(sentence, key=lambda w: abs(w["e"] - e))["e"]
    if duration:
        e = min(e, duration)
    s, e = round(s, 2), round(e, 2)
    if MIN_LEN <= e - s <= MAX_LEN:
        return dict(c, start=s, end=e)
    return None


def choose(cands, duration, n):
    """Pick n clips that don't overlap and come from different parts of the video.
    First the best strong clip from each slice of the video, then the best remaining clips as backups."""
    ranked = sorted(cands, key=lambda c: -c["final"])
    chosen = []

    def free(c):
        return all(c["end"] + GAP <= k["start"] or c["start"] >= k["end"] + GAP for k in chosen)

    slices = max(n, 1)
    for i in range(slices):
        lo, hi = duration * i / slices, duration * (i + 1) / slices
        for c in ranked:
            if lo <= c["start"] < hi and c["final"] >= MIN_FINAL and free(c):
                chosen.append(dict(c, strong=True))
                break
    for c in ranked:
        if len(chosen) >= n:
            break
        if free(c):
            chosen.append(dict(c, strong=False))
    return sorted(chosen, key=lambda c: -c["final"])


def pick(segments, words, duration, n, lang, sig, debug):
    ai = []
    try:
        if os.getenv("GEMINI_API_KEY"):
            debug["picker"] = "gemini"
            ai = pick_with_gemini(segments, duration, n + 3, lang)
        elif os.getenv("ANTHROPIC_API_KEY"):
            debug["picker"] = "claude"
            ai = pick_with_claude(segments, duration, n + 3, lang)
        else:
            debug["picker"] = "no API key, picture and sound only"
    except Exception as e:
        print("AI picker failed, using picture and sound only:", e)
        debug["picker"] = f"picture and sound only (AI error: {str(e)[:120]})"

    cands = []
    for c in ai:
        try:
            s, e = max(0.0, float(c["start"])), min(float(c["end"]), duration)
        except (KeyError, TypeError, ValueError):
            continue
        ai_score = 0.6 * num(c.get("hook_score")) + 0.4 * num(c.get("score"))
        cands.append({"start": s, "end": e, "final": 0.7 * ai_score + 0.3 * window_score(s, e, sig),
                      "title": str(c.get("title", ""))[:80], "reason": str(c.get("reason", "")) or "AI pick"})
    cands += signal_windows(duration, sig)

    starts = [w["s"] for w in words]
    ends = [w["e"] for w in words]
    snapped = [x for x in (snap_one(c, starts, ends, words, duration) for c in cands) if x]
    chosen = choose(snapped, duration, n)
    for k, c in enumerate(chosen):
        if not c["title"]:
            c["title"] = window_title(words, c["start"], c["end"]) or HYPE_TITLES[k % len(HYPE_TITLES)]
        if not c["strong"]:
            c["reason"] = "Backup pick. " + c["reason"]
    debug.update(ai_picks=len(ai), candidates=len(cands), usable=len(snapped),
                 strong_picks=sum(c["strong"] for c in chosen), clips=len(chosen))
    return chosen


def active_area(src, duration):
    """Find the real picture inside black letterbox bars so the bars don't end up in the Short."""
    times = [duration * f for f in (0.2, 0.5, 0.8)] if duration > 10 else [1.0]
    boxes = []
    for t in times:
        r = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "8",
                            "-vf", "cropdetect=24:16:0", "-f", "null", "-"], capture_output=True, text=True)
        found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
        if found:
            boxes.append(tuple(int(v) for v in found[-1]))
    if not boxes:
        return None
    x0 = min(b[2] for b in boxes)
    y0 = min(b[3] for b in boxes)
    x1 = max(b[2] + b[0] for b in boxes)
    y1 = max(b[3] + b[1] for b in boxes)
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    return (w, h, x0, y0) if w > 100 and h > 100 else None


def ts(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def clean_text(s):
    return re.sub(r"[{}\\]", "", s)


def caption_groups(ws, max_words=3, max_chars=22):
    """Split spoken words into short on-screen groups; break on long pauses."""
    groups, cur = [], []
    for w in ws:
        full = cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > 0.8
                        or sum(len(x["w"]) for x in cur) + len(w["w"]) > max_chars)
        if full:
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def is_power(word, lang):
    core = word.lower().strip(".,!?'\"")
    return core in HOOK_WORDS or (lang == "en" and len(core) >= 9)


def snap_to_beat(t, beats, window=0.15):
    if not beats:
        return t
    i = bisect.bisect_left(beats, t)
    near = [beats[j] for j in (i - 1, i) if 0 <= j < len(beats)]
    b = min(near, key=lambda x: abs(x - t))
    return b if abs(b - t) <= window else t


def karaoke(w, active, lang):
    text = clean_text(w["w"]).upper() if lang == "en" else clean_text(w["w"])
    if not active:
        return text
    if is_power(w["w"], lang):   # hook and long words: bigger pop, pink-red
        anim = "{\\c&H783CFF&\\t(0,90,\\fscx140\\fscy140)\\t(90,240,\\fscx112\\fscy112)}"
    else:                        # normal active word: yellow pop
        anim = "{\\c&H00E5FF&\\t(0,110,\\fscx118\\fscy118)\\t(110,200,\\fscx100\\fscy100)}"
    return anim + text + "{\\c&HFFFFFF&\\fscx100\\fscy100}"


def make_ass(words, start, end, title, path, lang, beats, credit=""):
    font = "Arial" if lang == "en" else CAPTION_FONT
    cap = 84 if lang in INDIC else 92
    length = end - start
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{cap},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,2,2,60,60,520,1
Style: Hook,{font},{cap - 18},&H00FFFFFF,&H00FFFFFF,&H005A28E6,&H00000000,1,0,0,0,100,100,0,0,3,14,0,8,70,70,230,1
Style: Credit,{font},44,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,4,0,2,60,60,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:  # hook title pops in on a colored box at the top
        lines.append(f"Dialogue: 1,{ts(0)},{ts(min(3.5, length))},Hook,,0,0,0,,"
                     "{\\fad(200,300)\\fscx60\\fscy60\\t(0,220,\\fscx100\\fscy100)}" + clean_text(title))
    if credit:  # small source credit at the bottom
        lines.append(f"Dialogue: 1,{ts(0)},{ts(length)},Credit,,0,0,0,,Source: {clean_text(credit)}")
    # thin red progress bar along the bottom that fills as the clip plays
    bar = f"{{\\an7\\pos(0,1904)\\bord0\\shad0\\c&H3C38E5&\\fscx1\\t(0,{int(length * 1000)},\\fscx100)\\p1}}"
    lines.append(f"Dialogue: 2,{ts(0)},{ts(length)},Cap,,0,0,0,,{bar}m 0 0 l 1080 0 1080 16 0 16{{\\p0}}")
    if CAPTIONS:
        ws = [w for w in words if start <= w["s"] < end]
        for grp in caption_groups(ws):
            gstart = snap_to_beat(grp[0]["s"], beats)
            if len(grp) > 1:
                gstart = min(gstart, grp[1]["s"] - 0.05)
            gstart = max(gstart, start)
            for j, w in enumerate(grp):
                s = gstart if j == 0 else w["s"]
                e = grp[j + 1]["s"] if j + 1 < len(grp) else w["e"]
                body = " ".join(karaoke(x, k == j, lang) for k, x in enumerate(grp))
                lines.append(f"Dialogue: 0,{ts(s - start)},{ts(max(e, s + 0.05) - start)},Cap,,0,0,0,,{body}")
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def render(src, d, i, c, words, lang, beats, area, credit):
    dur = c["end"] - c["start"]
    ass = f"clip_{i}.ass"
    make_ass(words, c["start"], c["end"], c.get("title", ""), d / ass, lang, beats, credit)
    zoom = f"1+0.05*t/{dur:.2f}+0.12*exp(-10*t)"            # slow push-in plus a hard pop at the very start
    if PUNCH_INS:
        zoom += "+0.07*exp(-9*(t-2*floor(t/2)))"             # punch-in every 2 seconds
    parts = [f"crop={area[0]}:{area[1]}:{area[2]}:{area[3]}"] if area else []  # remove letterbox bars
    parts += ["crop=ih*9/16:ih", "scale=1080:1920",
              f"scale=w='trunc(1080*({zoom})/2)*2':h='trunc(1920*({zoom})/2)*2':eval=frame",
              "crop=1080:1920",
              "eq=saturation=1.25:contrast=1.08:brightness=0.02",   # color boost
              "unsharp=5:5:0.8:5:5:0.0",                            # sharpen
              f"fade=t=in:st=0:d=0.15:color=white,fade=t=out:st={max(dur - 0.4, 0):.2f}:d=0.4",  # white flash in
              f"ass={ass}"]
    af = f"loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:st=0:d=0.2,afade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.5"
    r = subprocess.run(["ffmpeg", "-y", "-ss", f"{c['start']:.2f}", "-t", f"{dur:.2f}", "-i", src.name,
                        "-vf", ",".join(parts), "-af", af, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-movflags", "+faststart", f"clip_{i}.mp4"],
                       cwd=d, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("FFmpeg render failed: " + r.stderr[-400:])


def process(url, job, upd, n=None, language="auto"):
    n = n or CLIP_COUNT
    d = JOBS_DIR / job["id"]; d.mkdir(parents=True, exist_ok=True)
    debug = {}; job["debug"] = debug
    job["profile"] = {}
    step_start = time.time()
    upd("Downloading video", 8)
    src, duration, title, uploader = download(url, d)
    job["profile"]["download"] = (time.time() - step_start) * 1000  # ms
    job.update(video_title=title, duration=duration)
    step_start = time.time()
    upd("Listening to the speech (transcribing)", 22)
    audio = load_audio(src)
    segments, words, lang = transcribe(audio, language)
    job["profile"]["transcribe"] = (time.time() - step_start) * 1000  # ms
    job["detected_language"] = lang
    debug.update(language=lang, words_heard=len(words))
    step_start = time.time()
    upd("Scanning picture and sound for the wow moments", 40)
    sig = build_signals(src, audio, duration)
    beats = detect_beats(audio)
    job["profile"]["signal_build"] = (time.time() - step_start) * 1000  # ms
    job["beat_sync"] = bool(beats)
    step_start = time.time()
    upd("Finding the strongest hooks", 55)
    chosen = pick(segments, words, duration, n, lang, sig, debug)
    job["profile"]["pick"] = (time.time() - step_start) * 1000  # ms
    area = active_area(src, duration)
    print("Pipeline details:", debug)
    job["clips"] = []
    if not chosen:
        job["note"] = "No usable moments found. Details: " + ", ".join(f"{k} {v}" for k, v in debug.items())
        return
    # Render clips in parallel (limited workers to avoid OOM)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = min(4, os.cpu_count() or 2)
    futures = {}
    for k, c in enumerate(chosen, 1):
        upd(f"Rendering clip {k} of {len(chosen)} (effects + captions)", 60 + 38 * (k - 1) // len(chosen))
        futures[k] = (c, time.time())
    def render_task(idx, clip, start_time):
        try:
            render(src, d, idx, clip, words, lang, beats, area, uploader)
            job["profile"][f"render_{idx}"] = (time.time() - start_time) * 1000
            return (idx, None)
        except Exception as e:
            return (idx, e)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(render_task, idx, clip, st): idx for idx, (clip, st) in futures.items()}
        for fut in as_completed(future_to_idx):
            idx, err = fut.result()
            if err:
                print(f"Clip {idx} failed:", err)
    # End parallel rendering
        job["clips"].append({"id": k, "start": c["start"], "end": c["end"], "title": c["title"],
                             "score": round(c["final"]), "reason": c["reason"],
                             "file": f"/files/{job['id']}/clip_{k}.mp4"})
    if len(job["clips"]) < n:
        job["note"] = f"Made {len(job['clips'])} of {n} clips. Check the terminal for the ones that failed."