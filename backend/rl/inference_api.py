
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentinel_ppo_v4.zip"

model = PPO.load(str(MODEL_PATH), device="cpu")

app = FastAPI(title="SENTINEL PPO v4 Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

ACTIONS = {
    0: "APPROACH PLAYER",
    1: "APPROACH GUARDIAN",
    2: "ATTACK PLAYER",
    3: "ATTACK GUARDIAN",
    4: "RETREAT",
}


class PredictionRequest(BaseModel):
    observation: list[float] = Field(min_length=15, max_length=15)


@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "sentinel_ppo_v4",
        "observations": 15,
        "actions": 5,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    obs = np.asarray(request.observation, dtype=np.float32)

    if not np.all(np.isfinite(obs)):
        raise HTTPException(
            status_code=422,
            detail="Observation contains invalid numbers",
        )

    if np.any(obs < -1) or np.any(obs > 1):
        raise HTTPException(
            status_code=422,
            detail="Observation values must be between -1 and 1",
        )

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    return {
        "action": action,
        "decision": ACTIONS[action],
        "model": "sentinel_ppo_v4",
    }
