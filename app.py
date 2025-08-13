# app.py

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from utils.config_handler import Config
from fastAPI_test import run_training, training_status

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h1>Mineral Recognition Training</h1>
        <p><a href="/docs">Go to API docs</a></p>
        <form action="/train" method="post">
            <button type="submit">Start Training</button>
        </form>
    </body>
    </html>
    """


# GET CONFIG
@app.get("/config")
def get_config():
    config = Config.from_file()
    return config.__dict__


# UPDATE CONFIG
@app.post("/config")
def update_config(new_values: dict):
    config = Config.from_file()
    for key, value in new_values.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.save()
    return {"status": "Config updated"}


# START TRAINING
@app.post("/train")
def start_training(background_tasks: BackgroundTasks):
    if training_status["running"]:
        return {"status": "Training already running"}

    config = Config.from_file()
    background_tasks.add_task(run_training, config)
    return {"status": "Training started"}


# TRAINING STATUS
@app.get("/status")
def get_status():
    return training_status
