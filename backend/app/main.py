import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .fixtures import fetch_upcoming
from .model import (
    get_classes,
    load_features,
    load_model,
    predict_by_teams,
)
from .schemas import (
    HealthResponse,
    PredictionResponse,
    TeamMatchRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_features()
    try:
        await asyncio.to_thread(fetch_upcoming)
    except Exception:
        pass  # don't block startup if the API is down
    yield


app = FastAPI(
    title="EPL Match Outcome Predictor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


NO_FIXTURES_MESSAGE = "Match fixtures are still to be decided."


@app.get("/upcoming")
def upcoming():
    try:
        fixtures = fetch_upcoming()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fixture fetch failed: {e}")

    if not fixtures:
        return {"count": 0, "message": NO_FIXTURES_MESSAGE, "fixtures": []}

    out = []
    for fx in fixtures:
        item = dict(fx)
        try:
            pred = predict_by_teams(fx["home"], fx["away"])
            item["model"] = pred
            if fx["market"]:
                item["edge"] = {
                    k: round(pred["probabilities"].get(k, 0) - fx["market"][k], 4)
                    for k in ("H", "D", "A")
                }
        except Exception as e:
            item["error"] = str(e)
        out.append(item)

    return {"count": len(out), "message": None, "fixtures": out}

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        classes = get_classes()
        load_features()
        return {
            "status": "ok",
            "model_loaded": True,
            "features_loaded": True,
            "classes": classes,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict-by-teams", response_model=PredictionResponse)
def predict_teams(body: TeamMatchRequest):
    try:
        return predict_by_teams(body.home_team, body.away_team)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "EPL Outcome Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict_by_teams": "POST /predict-by-teams",
        "upcoming": "GET /upcoming",
    }