import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, log_df
from src.io.api import load_apikeys
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from src.feature.cross_section import CrossSectionManager
from src.utils.df import transpose_df

#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 


limit = 5
t0 = log_info("Initialise")
api_keys = load_apikeys()
history_start = "1980-01-01"
asset_type = "equities"
filters = {"currency": ['usd'], "exchange": ['nms', 'nyq', 'ngm', 'ncm', 'ase', 'pcx', 'pnk'], "market_cap": ["mega_cap"]}

dm = DatabaseManager() 
database = dm.fetch_database(asset_type=asset_type, filters=filters) #fix asset_type so derived instead
database = dm.equity_filter(database, asset_type)
database = database.limit(limit).collect()
log_df(database, "DM loaded")
print("-"*120)
log_info("Initialise", t0)


t0 = log_info("Asset")
frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None), "yahooquery": None}
pm = PriceManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
history, failed_df = pm.load_history(asset_type, database, frequency, refresh_threshold_days=2, REFRESH=False) #threshold
#log_df(history, "PM loaded", verbose=3)
log_df(failed_df, "PM excludes")
print("-"*120)

frequency = "annual"
sources = {"yahooquery": None}
fm = FundamentalManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
fa, failed_df = fm.load_fundamentals(asset_type, database, frequency=frequency, refresh_threshold_days=None, REFRESH=False)
#log_df(fa, "FM loaded")
log_df(failed_df, "FM excludes")
print("-"*120)
log_info("Asset", t0)


t0 = log_info("Benchmark")
asset_type = "etfs"
benchmarks = ["ACWI", "SPY", "GLD"]
database = dm.fetch_database(asset_type=asset_type, filters=None) #fix asset_type so derived instead
database = dm._reader.filter_lazy(database, filters={"symbol": benchmarks})
database = database.collect()
log_df(database, "DM loaded")
print("-"*120)

frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None), "yahooquery": None}
pm = PriceManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
benchmark, failed_df = pm.load_history(asset_type, database, frequency, refresh_threshold_days=2, REFRESH=False) #threshold
#log_df(benchmark, "BM loaded", verbose=3)
log_df(failed_df, "BM excludes")
print("-"*120)
log_info("Benchmark", t0)


t0 = log_info("Cross Section")
cm = CrossSectionManager()
rv = cm.get_relative_valuation(fa, history, benchmark, n_dates=5)
log_df(rv, "RV loaded")
rv.write_csv("rv.csv")
print("-"*120)
log_info("Cross Section", t0)





        






