from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import numpy as np
from uni2ts.model.moirai import MoiraiModule, MoiraiFinetune
import os
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
df = None

@app.on_event("startup")
async def load_model():
    global model, df
    print("Loading Data...")
    if os.path.exists("data/processed/omni_processed.csv"):
        df = pd.read_csv("data/processed/omni_processed.csv", parse_dates=["Datetime"], index_col="Datetime")
    
    print("Loading Moirai Model...")
    # Find the best checkpoint
    ckpt_dir = "models/moirai_checkpoints"
    if os.path.exists(ckpt_dir):
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if checkpoints:
            best_ckpt = sorted(checkpoints)[0] # Grab the first one
            try:
                module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-small')
                model = MoiraiFinetune.load_from_checkpoint(os.path.join(ckpt_dir, best_ckpt), module=module)
                model.eval()
                print(f"Loaded checkpoint: {best_ckpt}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                model = None
        else:
            print("No checkpoints found yet! API will run in mock mode.")
    else:
        print("Checkpoint directory not found! API will run in mock mode.")

@app.get("/api/forecast")
async def get_forecast():
    """
    Returns historical data and 12-hour forecast for Pc5 Wave Power.
    For hackathon demo, we will return the last 24 hours of data + 12 hours of forecast.
    """
    current_time = pd.Timestamp.now()
    
    history = []
    for i in range(24):
        t = current_time - pd.Timedelta(hours=24-i)
        # Mock historical wave power (normal baseline)
        val = 2.0 + math.sin(i / 3.0) + np.random.normal(0, 0.2)
        history.append({"time": t.strftime("%H:%M"), "Pc5_Wave_Power": max(0, val), "type": "history"})
        
    forecast = []
    for i in range(12):
        t = current_time + pd.Timedelta(hours=i)
        # Mock forecast (simulating a Solar Storm spike in 5 hours)
        base = 2.0 + math.sin((24+i) / 3.0)
        spike = 15.0 if 4 <= i <= 6 else 0
        val = base + spike + np.random.normal(0, 0.5)
        forecast.append({"time": t.strftime("%H:%M"), "Pc5_Wave_Power": max(0, val), "type": "forecast"})
        
    return {"history": history, "forecast": forecast}
