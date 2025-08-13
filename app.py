# app.py
# FASTAPI BACKEND THAT:
# - SERVES A STATIC HTML/CSS/JS GUI
# - EXPOSES /config (GET/POST), /upload-config (POST), /download-config (GET)
# - EXPOSES /train (POST), /status (GET), /results (GET)

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict
from fastapi.responses import JSONResponse
from utils.config_handler import Config, DEFAULT_CONFIG
from spectrum_network_train import run_training, training_status
import os

app = FastAPI()

# SERVE STATIC GUI (static/index.html, static/styles.css, static/script.js)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


# -------------------- CONFIG ENDPOINTS --------------------

@app.get("/config")
def get_config() -> Dict:
    cfg = Config.from_file()
    # return only known keys (avoid private attrs)
    return {k: getattr(cfg, k) for k in DEFAULT_CONFIG.keys()}

@app.post("/config")
async def update_config(payload: Dict):
    cfg = Config.from_file()
    for k, v in payload.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.save()
    return {"status": "ok"}


@app.post("/upload-config")
async def upload_config(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt config files are accepted.")

    content = (await file.read()).decode("utf-8", errors="replace")
    # Parse the uploaded text using the same rules your Config.from_file uses.
    loaded = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in DEFAULT_CONFIG:
            loaded[key] = Config._parse_value(key, value)  # reuse your parser

    # Fill defaults for missing keys
    for key, default in DEFAULT_CONFIG.items():
        loaded.setdefault(key, default)

    # Save to the main config.txt
    cfg = Config(**loaded)
    cfg.save()
    return {"status": "ok", "message": f"Loaded {file.filename} into config.txt"}


@app.get("/download-config", response_class=PlainTextResponse)
def download_config():
    cfg = Config.from_file()
    # Return as downloadable text (browser will download due to content-disposition)
    lines = []
    for k in DEFAULT_CONFIG.keys():
        lines.append(f"{k}={getattr(cfg, k)}")
    text = "\n".join(lines)
    headers = {"Content-Disposition": 'attachment; filename="config.txt"'}
    return PlainTextResponse(text, headers=headers)


# -------------------- TRAINING & STATUS --------------------

@app.post("/train")
def start_training(background_tasks: BackgroundTasks):
    if training_status.get("running"):
        return {"status": "already_running"}

    cfg = Config.from_file()
    background_tasks.add_task(run_training, cfg)
    return {"status": "started"}

@app.get("/status")
def get_status():
    return training_status

@app.get("/training_status")
def get_training_status():
    """
    Returns the current training status for progress tracking.
    Should include current epoch, total epochs, loss, val_loss, running state.
    """
    response = {
        "running": training_status.get("running", False),
        "epoch": training_status.get("epoch", 0),
        "loss": training_status.get("loss"),
        "val_loss": training_status.get("val_loss"),
        "total_epochs": training_status.get("total_epochs", 0)
    }
    return JSONResponse(content=response)

@app.get("/results")
def get_results():
    # We expose final metrics once training_status['running'] is False
    if training_status.get("running", True):
        return {"ready": False}

    # Accuracy (float) and confusion matrix (2D list) are filled by run_training()
    acc = training_status.get("final_accuracy", None)
    cm = training_status.get("confusion_matrix", None)
    return {"ready": True, "accuracy": acc, "confusion_matrix": cm}
    acc = training_status.get("final_accuracy", None)
    cm = training_status.get("confusion_matrix", None)
    return {"ready": True, "accuracy": acc, "confusion_matrix": cm}
