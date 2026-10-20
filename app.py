import threading, traceback, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pipeline

pipeline.JOBS_DIR.mkdir(exist_ok=True)
app = FastAPI(title="AI Video Clipper API", version="1.0.0")
app.mount("/files", StaticFiles(directory=pipeline.JOBS_DIR), name="files")
JOBS = {}

class Req(BaseModel):
    url: str

def run(job):
    try:
        pipeline.process(job["url"], job, lambda s, p: job.update(stage=s, progress=p))
        job.update(status="done", stage="Ready to post", progress=100)
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=str(e)[:300])

@app.post("/api/analyze")
def analyze(r: Req):
    if "youtu" not in r.url:
        raise HTTPException(400, "Paste a valid YouTube link")
    job = {"id": uuid.uuid4().hex[:10], "url": r.url, "status": "running", "stage": "Starting", "progress": 2, "clips": []}
    JOBS[job["id"]] = job
    threading.Thread(target=run, args=(job,), daemon=True).start()
    return {"id": job["id"]}

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[jid]

@app.get("/")
def index():
    return FileResponse("static/index.html")
