from pathlib import Path
from dotenv import dotenv_values
import datetime


def load_apikeys(env_path=None):
    if env_path is None:
        env_path = Path.cwd() / 'config' / '.env'
    if not env_path.exists():
        print(f"[ERROR] No file at: {env_path.absolute()}")
        return {}
    config = dotenv_values(str(env_path))
    return dict(config)

def fetch_dates(n=3):
    today = datetime.date.today()
    year = today.year

    past_dates = []
    y = year
    while len(past_dates) < n and y > 0:
        d = datetime.date(year=y, month=4, day=3)
        if d <= today:
            past_dates.append(d)
        y -= 1
    past_dates = sorted(past_dates)
    return past_dates + [today]

def validate_date(startDate):
    try:
        datetime.datetime.strptime(startDate, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid format '{startDate}', expected YYYY-MM-DD")

def get_q_dates(year):
    assert isinstance(year, (int, str)), "year has invalid data type"
    year = int(year)
    return {
        "Q1": "{year}-04-15:{year}-05-15",
        "Q2": "{year}-07-15:{year}-08-15",
        "Q3": "{year}-10-15:{year}-11-15",
        "Q4": "{year+1}-01-15:{year+1}-02-15"
    }



















