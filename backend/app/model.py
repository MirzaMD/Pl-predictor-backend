from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from .config import (
    MODEL_PATH,
    FEATURES_PATH,
    TEAM_NAME_MAP,
    OUTCOME_LABELS,
)

_model_bundle: dict[str, Any] | None = None
_features_df: pd.DataFrame | None = None
_feature_columns: list[str] | None = None


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    global _model_bundle, _feature_columns
    if _model_bundle is None:
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        _model_bundle = joblib.load(path)
        _feature_columns = list(
            _model_bundle.get("feature_columns")
            or []
        )
    return _model_bundle


def get_model():
    return load_model()["model"]


def get_classes() -> list[str]:
    bundle = load_model()
    return list(bundle.get("classes") or get_model().classes_)


def get_feature_columns() -> list[str]:
    load_model()
    if not _feature_columns:
        raise RuntimeError("feature_columns missing from model bundle")
    return _feature_columns


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    global _features_df
    if _features_df is None:
        if not path.exists():
            raise FileNotFoundError(f"Features CSV not found at {path}")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values(["Date", "MatchID"]).reset_index(drop=True)
        _features_df = df
    return _features_df


def normalize_team_name(name: str) -> str:
    name = (name or "").strip()
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # already in training form
    df = load_features()
    known = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    if name in known:
        return name
    raise ValueError(
        f"Unknown team name: {name!r}. "
        f"Add a mapping in TEAM_NAME_MAP."
    )


def _latest_home_row(team: str) -> pd.Series:
    df = load_features()
    rows = df[df["HomeTeam"] == team]
    if rows.empty:
        raise ValueError(f"No home history for team {team!r}")
    return rows.iloc[-1]


def _latest_away_row(team: str) -> pd.Series:
    df = load_features()
    rows = df[df["AwayTeam"] == team]
    if rows.empty:
        raise ValueError(f"No away history for team {team!r}")
    return rows.iloc[-1]


def _expected_from_elo(home_elo: float, away_elo: float) -> tuple[float, float]:
    """Standard Elo expected score (no extra home boost beyond training)."""
    diff = home_elo - away_elo
    exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    return exp_home, 1.0 - exp_home


def build_feature_row(home_team: str, away_team: str) -> dict:
    """
    Build one model input row from latest known form/Elo
    for home (as home) and away (as away).
    """
    home = normalize_team_name(home_team)
    away = normalize_team_name(away_team)

    h = _latest_home_row(home)
    a = _latest_away_row(away)

    home_elo = float(h["Home_Elo"])
    away_elo = float(a["Away_Elo"])
    elo_diff = home_elo - away_elo
    exp_home, exp_away = _expected_from_elo(home_elo, away_elo)

    row = {
        # Home form / venue
        "Home_MatchesPlayedPrior": h["Home_MatchesPlayedPrior"],
        "Home_PointsAvg5": h["Home_PointsAvg5"],
        "Home_GoalsForAvg5": h["Home_GoalsForAvg5"],
        "Home_GoalsAgainstAvg5": h["Home_GoalsAgainstAvg5"],
        "Home_GoalDifferenceAvg5": h["Home_GoalDifferenceAvg5"],
        "Home_WinRate5": h["Home_WinRate5"],
        "Home_DrawRate5": h["Home_DrawRate5"],
        "Home_LossRate5": h["Home_LossRate5"],
        "Home_VenueMatchesPrior": h["Home_VenueMatchesPrior"],
        "Home_VenuePointsAvg5": h["Home_VenuePointsAvg5"],
        "Home_VenueGoalsForAvg5": h["Home_VenueGoalsForAvg5"],
        "Home_VenueGoalsAgainstAvg5": h["Home_VenueGoalsAgainstAvg5"],
        "Home_VenueGoalDifferenceAvg5": h["Home_VenueGoalDifferenceAvg5"],
        # Away form / venue
        "Away_MatchesPlayedPrior": a["Away_MatchesPlayedPrior"],
        "Away_PointsAvg5": a["Away_PointsAvg5"],
        "Away_GoalsForAvg5": a["Away_GoalsForAvg5"],
        "Away_GoalsAgainstAvg5": a["Away_GoalsAgainstAvg5"],
        "Away_GoalDifferenceAvg5": a["Away_GoalDifferenceAvg5"],
        "Away_WinRate5": a["Away_WinRate5"],
        "Away_DrawRate5": a["Away_DrawRate5"],
        "Away_LossRate5": a["Away_LossRate5"],
        "Away_VenueMatchesPrior": a["Away_VenueMatchesPrior"],
        "Away_VenuePointsAvg5": a["Away_VenuePointsAvg5"],
        "Away_VenueGoalsForAvg5": a["Away_VenueGoalsForAvg5"],
        "Away_VenueGoalsAgainstAvg5": a["Away_VenueGoalsAgainstAvg5"],
        "Away_VenueGoalDifferenceAvg5": a["Away_VenueGoalDifferenceAvg5"],
        # Elo
        "Home_Elo": home_elo,
        "Away_Elo": away_elo,
        "Elo_Difference": elo_diff,
        "Expected_Home_Result": exp_home,
        "Expected_Away_Result": exp_away,
        # Relative
        "Points_Difference": (
            float(h["Home_PointsAvg5"]) - float(a["Away_PointsAvg5"])
            if pd.notna(h["Home_PointsAvg5"]) and pd.notna(a["Away_PointsAvg5"])
            else np.nan
        ),
        "GoalDifference_Difference": (
            float(h["Home_GoalDifferenceAvg5"]) - float(a["Away_GoalDifferenceAvg5"])
            if pd.notna(h["Home_GoalDifferenceAvg5"]) and pd.notna(a["Away_GoalDifferenceAvg5"])
            else np.nan
        ),
        "Venue_GoalDifference_Difference": (
            float(h["Home_VenueGoalDifferenceAvg5"]) - float(a["Away_VenueGoalDifferenceAvg5"])
            if pd.notna(h["Home_VenueGoalDifferenceAvg5"]) and pd.notna(a["Away_VenueGoalDifferenceAvg5"])
            else np.nan
        ),
        # Teams
        "HomeTeam": home,
        "AwayTeam": away,
    }
    return row


def predict_match(features: dict) -> dict:
    model = get_model()
    classes = get_classes()
    cols = get_feature_columns()

    # Keep only columns the model knows; order matters
    row = {col: features.get(col) for col in cols}
    X = pd.DataFrame([row], columns=cols)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    probabilities = {str(c): float(p) for c, p in zip(classes, proba)}

    return {
        "predicted_outcome": str(pred),
        "predicted_label": OUTCOME_LABELS.get(str(pred), str(pred)),
        "probabilities": probabilities,
        "home_team": features.get("HomeTeam", ""),
        "away_team": features.get("AwayTeam", ""),
    }


def predict_by_teams(home_team: str, away_team: str) -> dict:
    features = build_feature_row(home_team, away_team)
    return predict_match(features)