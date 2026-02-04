import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.manager import ManagerManager
from src.io.parquet import ParquetManager
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from config.config import DATABASE_CONFIGS


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 
#On demand vs Cached: ?(ml | offline)


t0 = log_info("Fetch")
api_keys = load_apikeys()
pm = ParquetManager()
mm = ManagerManager(pm)

limit = 1 #dry run
history_start = "1980-01-01"
asset_type = DATABASE_CONFIGS["asset_type"]

dm = DatabaseManager(pm=pm)
database = dm.fetch_database(**DATABASE_CONFIGS, limit=None, COLLECT=True)
log_df(database, "DM loaded")
print("-"*120)

frequency = "annual"
sources = {"yahooquery": None}
database, exclude = dm.filter_excludes(database.lazy(), asset_type, filters={"asset_type": asset_type, "source": list(sources.keys())}, limit=limit, COLLECT=True)
fm = FundamentalManager(mm=mm, sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
df, partition_cols, failed_df = fm.load_history(asset_type, database, frequency=frequency, refresh_threshold_days=None, REFRESH=False)
log_df(df, "FM loaded")
log_df(failed_df, "FM excludes")
print(f"Partitioned by: {partition_cols}")
print("-"*120)

"""
frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
database, exclude = dm.filter_excludes(database.lazy(), asset_type, filters={"asset_type": asset_type, "source": list(sources.keys())}, limit=limit, COLLECT=True)
pm = PriceManager(mm=mm, sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
df, partition_cols, failed_df = pm.load_history(asset_type, database, frequency=frequency, refresh_threshold_days=None, REFRESH=False)
log_df(df, "PM loaded")
log_df(failed_df, "PM excludes")
print(f"Partitioned by: {partition_cols}")
print("-"*120)

log_info("Fetch", t0) 
"""
