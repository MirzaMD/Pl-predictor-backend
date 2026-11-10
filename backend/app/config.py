from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "epl_outcome_rf.joblib"
FEATURES_PATH = BASE_DIR / "PL_Feature_Engineered_Elo.csv"

# API / display name → name used in your training data
TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton &amp; Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "West Ham United": "West Ham",
    "Crystal Palace": "Crystal Palace",
    "Sheffield United": "Sheffield United",
    "Luton Town": "Luton",
    "Norwich City": "Norwich",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",
    "Leeds United": "Leeds",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Everton": "Everton",
    "Aston Villa": "Aston Villa",
    "Southampton": "Southampton",
    "Burnley": "Burnley",
    "Watford": "Watford",
    "Sunderland": "Sunderland",
    "Hull City": "Hull",
    "Hull": "Hull",
    "Middlesbrough": "Middlesbrough",
    "Coventry City": "Coventry",
    "Coventry": "Coventry",
}

OUTCOME_LABELS = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}