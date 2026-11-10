from pydantic import BaseModel, Field
from typing import Dict, Optional

class MatchFeatures(BaseModel):
    # Home
    Home_MatchesPlayedPrior: float = Field(..., ge=0)
    Home_PointsAvg5: Optional[float] = None
    Home_GoalsForAvg5: Optional[float] = None
    Home_GoalsAgainstAvg5: Optional[float] = None
    Home_GoalDifferenceAvg5: Optional[float] = None
    Home_WinRate5: Optional[float] = None
    Home_DrawRate5: Optional[float] = None
    Home_LossRate5: Optional[float] = None
    Home_VenueMatchesPrior: float = Field(..., ge=0)
    Home_VenuePointsAvg5: Optional[float] = None
    Home_VenueGoalsForAvg5: Optional[float] = None
    Home_VenueGoalsAgainstAvg5: Optional[float] = None
    Home_VenueGoalDifferenceAvg5: Optional[float] = None

    # Away
    Away_MatchesPlayedPrior: float = Field(..., ge=0)
    Away_PointsAvg5: Optional[float] = None
    Away_GoalsForAvg5: Optional[float] = None
    Away_GoalsAgainstAvg5: Optional[float] = None
    Away_GoalDifferenceAvg5: Optional[float] = None
    Away_WinRate5: Optional[float] = None
    Away_DrawRate5: Optional[float] = None
    Away_LossRate5: Optional[float] = None
    Away_VenueMatchesPrior: float = Field(..., ge=0)
    Away_VenuePointsAvg5: Optional[float] = None
    Away_VenueGoalsForAvg5: Optional[float] = None
    Away_VenueGoalsAgainstAvg5: Optional[float] = None
    Away_VenueGoalDifferenceAvg5: Optional[float] = None

    # Elo
    Home_Elo: float
    Away_Elo: float
    Elo_Difference: float
    Expected_Home_Result: float = Field(..., ge=0, le=1)
    Expected_Away_Result: float = Field(..., ge=0, le=1)

    # Relative
    Points_Difference: float
    GoalDifference_Difference: float
    Venue_GoalDifference_Difference: float

    # Teams
    HomeTeam: str
    AwayTeam: str

class TeamMatchRequest(BaseModel):
    home_team: str = Field(..., examples=["Crystal Palace"])
    away_team: str = Field(..., examples=["Liverpool"])


class PredictionResponse(BaseModel):
    predicted_outcome: str
    predicted_label: str
    probabilities: Dict[str, float]
    home_team: str
    away_team: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_loaded: bool
    classes: list[str]

class FixturePrediction(BaseModel):
    match_id: int
    kickoff: str
    status: str
    home: str
    away: str
    score: Optional[Dict[str, int]] = None
    model: Optional[PredictionResponse] = None
    market: Optional[Dict[str, float]] = None
    edge: Optional[Dict[str, float]] = None   # model minus market
    error: Optional[str] = None

class TodayResponse(BaseModel):
    date: str
    count: int
    fixtures: list[FixturePrediction]