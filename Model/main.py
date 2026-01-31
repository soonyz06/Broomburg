import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, df_display
from src.io.fetch import load_apikeys
from src.io.parquet import ParquetManager
from src.io.database import DatabaseManager
from src.io.price import PriceManager
from src.io.sec import SECManager


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple


t0 = log_info("Fetch")
api_keys = load_apikeys()
io = ParquetManager()

asset_type = "equities" 
filters = {
    "currency": ['USD'],
    "exchange": ['NMS', 'NYQ', 'NGM', 'NCM', 'ASE', 'PCX', 'PNK'],
    "market_cap": ["Mega Cap"]
}
limit = 3 #dry run


dm = DatabaseManager(pm=io)
df, exclude = dm.filter_database(asset_type, filters=filters, limit=limit, REFRESH=False)
df_display(df, "DM loaded")
df_display(exclude, "DM excludes")


"""
sources = {"edgar": None}
sm = SECManager(pm=io, sources=sources)
df, failed_df = sm.load_sec(df, REFRESH=False)
df_display(df, "SM loaded")
df_display(failed_df, "SM excludes")



"""
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
pm = PriceManager(pm=io, sources=sources)
df, failed_df = pm.load_history(asset_type, df, REFRESH=False)
df_display(df, "PM loaded")


log_info("Fetch", t0)

##raw: fred, oecd, databento, newspaper+magazines
##load data -> read, eda, cleaning, preprocessing, feature construction, feature selection, modelling
#plotly dash





