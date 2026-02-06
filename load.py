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
from src.feature.preprocessing import CrossSectionManager #
from config.config import DATABASE_CONFIGS


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 
#On demand vs Cached: ?(ml | offline)


t0 = log_info("Initialise")
asset_type = DATABASE_CONFIGS["asset_type"]
history_start = "1980-01-01"
api_keys = load_apikeys()
pm = ParquetManager() #clean up one day
mm = ManagerManager(pm)
dm = DatabaseManager(pm=pm) 
database = dm.fetch_database(**DATABASE_CONFIGS, limit=None, COLLECT=False)
database = dm.equity_filter(database, asset_type, COLLECT=False)
log_df(database.collect(), "DM loaded")

frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
pm = PriceManager(mm=mm, sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
print("-"*120)
log_info("Initialise", t0) 


limit = 10 #dry run

t0 = log_info("Fetch")
frequency = "annual"
sources = {"yahooquery": None}
df, exclude = dm.filter_excludes(database, asset_type, filters={"asset_type": asset_type, "source": list(sources.keys())}, limit=limit, COLLECT=True)
fm = FundamentalManager(mm=mm, sources=sources, history_start=history_start, pm=pm)#.compact_job(asset_type, frequency=frequency)
df, failed_df = fm.load_fundamentals(asset_type, df, frequency=frequency, refresh_threshold_days=None, REFRESH=False)
log_df(df, "FM loaded")
log_df(failed_df, "FM excludes")
df.write_csv("hello.csv")
print("-"*120)



cm = CrossSectionManager(target="MC") 
fa = cm.to_cross_section(df, n_dates=5, n_symbols=None)
log_df(fa, "FA loaded")

log_info("Fetch", t0) 

##check refresh_threshold_days of price


        






