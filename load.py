import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from src.ml.feature.preprocessing import CrossSectionManager #
from config.config import DATABASE_CONFIGS


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 
#On demand vs Cached: ?(ml | offline)


t0 = log_info("Initialise")
asset_type = DATABASE_CONFIGS["asset_type"]
history_start = "1980-01-01"

api_keys = load_apikeys()
dm = DatabaseManager() 
database = dm.fetch_database(**DATABASE_CONFIGS, limit=None, COLLECT=False)
database = dm.equity_filter(database, asset_type, COLLECT=False)
log_df(database.collect(), "DM loaded")
print("-"*120)
log_info("Initialise", t0) 


limit = 3 #dry run


t0 = log_info("Fetch")
frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
pm = PriceManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
df, exclude = dm.filter_excludes(database, asset_type, filters={"asset_type": asset_type, "source": list(sources.keys())}, limit=limit, COLLECT=True)
df, failed_df = pm.load_history(asset_type, df, frequency, refresh_threshold_days=None, REFRESH=False)
log_df(df, "PM loaded")
log_df(failed_df, "PM excludes")


frequency = "annual"
sources = {"yahooquery": None}
df, exclude = dm.filter_excludes(database, asset_type, filters={"asset_type": asset_type, "source": list(sources.keys())}, limit=limit, COLLECT=True)
fm = FundamentalManager(sources=sources, history_start=history_start, pm=pm)#.compact_job(asset_type, frequency=frequency)
fa, failed_df = fm.load_fundamentals(asset_type, df, frequency=frequency, refresh_threshold_days=None, REFRESH=False)
rv = fm._yq.get_relative_valuation(fa)
dfs = fm._yq.get_tables(fa, rv)
for symbol, data in dfs.items():
    for key, df in data.items():
        log_df(df, f"{key} loaded", verbose=1)
print("-"*120)



cm = CrossSectionManager(target="MC") 
fa = cm.to_cross_section(fa, n_dates=5, n_symbols=None)
log_df(fa, "FA loaded")

log_info("Fetch", t0) 

##check refresh_threshold_days of price


        






