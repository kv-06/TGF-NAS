"""
FastAPI backend for HAR Pipeline
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import tempfile
import threading
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HAR Pipeline API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: dict     = {}
job_logs: dict = {}
job_dirs: dict = {}


class LogCapture:
    def __init__(self, job_id, original):
        object.__setattr__(self, '_job_id',   job_id)
        object.__setattr__(self, '_original', original)

    def write(self, msg):
        orig   = object.__getattribute__(self, '_original')
        job_id = object.__getattribute__(self, '_job_id')
        if msg.strip():
            job_logs[job_id].append(msg.rstrip())
        orig.write(msg)

    def flush(self):
        object.__getattribute__(self, '_original').flush()

    def isatty(self):
        return getattr(object.__getattribute__(self, '_original'), 'isatty', lambda: False)()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_original'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_original'), name, value)


def run_pipeline_job(job_id, train_path, test_path, input_size, output_size,
                     use_embedding, num_architectures, sparsity, quant_types,
                     dataset_fraction, save_dir):
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = LogCapture(job_id, orig_out)
    sys.stderr = LogCapture(job_id, orig_err)
    try:
        jobs[job_id]["status"] = "running"
        from HAR import run_pipeline
        result = run_pipeline(
            train_path=train_path, test_path=test_path,
            input_size=input_size, output_size=output_size,
            use_embedding=use_embedding, num_architectures=num_architectures,
            sparsity=sparsity, quant_types=quant_types,
            dataset_fraction=dataset_fraction, save_dir=save_dir,
        )
        jobs[job_id]["status"]      = "done"
        jobs[job_id]["model_paths"] = result.get("model_paths", {})
        jobs[job_id]["result"] = {
            "accuracy":                result.get("accuracy"),
            "sparsity":                result.get("sparsity"),
            "quantization":            result.get("quantization"),
            "best_architecture_index": result.get("best_architecture_index"),
            "stage_metrics":           result.get("stage_metrics"),
        }
    except Exception:
        import traceback
        err = traceback.format_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = err
        job_logs[job_id].append(f"\n[ERROR] {err}")
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        for p in [train_path, test_path]:
            try: os.remove(p)
            except Exception: pass


@app.post("/run")
async def start_pipeline(
    background_tasks:  BackgroundTasks,
    train_file:        UploadFile = File(...),
    test_file:         UploadFile = File(...),
    input_size:        int   = Form(0),
    output_size:       int   = Form(0),
    use_embedding:     bool  = Form(False),
    num_architectures: int   = Form(20),
    sparsity:          float = Form(0.3),
    quant_types:       str   = Form("FP16,INT8"),
    dataset_fraction:  float = Form(1.0),
):
    job_id   = str(uuid.uuid4())
    tmpdir   = tempfile.mkdtemp()
    save_dir = os.path.join(tmpdir, "models")
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(tmpdir, "train.csv")
    test_path  = os.path.join(tmpdir, "test.csv")
    with open(train_path, "wb") as f: f.write(await train_file.read())
    with open(test_path,  "wb") as f: f.write(await test_file.read())

    quant_list       = [q.strip() for q in quant_types.split(",") if q.strip()]
    dataset_fraction = max(0.05, min(1.0, dataset_fraction))

    jobs[job_id]     = {"status": "queued", "result": None, "error": None, "model_paths": {}}
    job_logs[job_id] = []
    job_dirs[job_id] = save_dir

    threading.Thread(
        target=run_pipeline_job,
        args=(job_id, train_path, test_path, input_size, output_size,
              use_embedding, num_architectures, sparsity, quant_list,
              dataset_fraction, save_dir),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job  = jobs[job_id]
    resp = {"job_id": job_id, "status": job["status"],
            "log_lines": len(job_logs.get(job_id, []))}
    if job["status"] == "done":
        resp["result"]      = job["result"]
        resp["model_paths"] = list(job.get("model_paths", {}).keys())
    elif job["status"] == "error":
        resp["error"] = job["error"]
    return resp


@app.get("/logs/{job_id}")
def get_logs(job_id: str, from_line: int = 0):
    if job_id not in job_logs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    all_logs = job_logs[job_id]
    return {"job_id": job_id, "from_line": from_line,
            "lines": all_logs[from_line:], "total_lines": len(all_logs),
            "status": jobs.get(job_id, {}).get("status", "unknown")}


@app.get("/download/{job_id}/{model_name}")
def download_model(job_id: str, model_name: str):
    """
    Download a saved .pth checkpoint.
    model_name: full_trained | pruned | quant_fp16 | quant_int8
    """
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if jobs[job_id]["status"] != "done":
        return JSONResponse(status_code=400, content={"error": "Pipeline not finished yet"})
    model_paths = jobs[job_id].get("model_paths", {})
    if model_name not in model_paths:
        return JSONResponse(status_code=404,
            content={"error": f"'{model_name}' not found. Available: {list(model_paths.keys())}"})
    path = model_paths[model_name]
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": f"File missing on disk: {path}"})
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"har_{model_name}.pth")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)