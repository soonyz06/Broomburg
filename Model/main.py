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
"""

##Check: Compact job, md, ifrs(nvo, pdd)
log_info("Fetch", t0)


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, offline
#Coupling, Locality

RAW={
    "macro": ["fred", "oecd", "wb"],
    "news": ["rss", "guardian", "8k", "s-1"],
    "analyst": ["ratings", "earnings", "estimates"],
    "alt": ["google trends", "3/4/5", "13F/D/G"],
    "high-level": ["databento"]
     }         

APP = {
    "productivity": {
        {"notes": ["adjacency list with set", "free-moving network graph with forces (center, repel and link)"]},
        {"todo": ["priority", "calender", "recurring"]},
        {"clock": ["timer", "alarm"]},
        {"storage": ["upload", "download", "query"]},
        {"inbox": []},
        {"bookmark": []}
    }
    "financail": {
        {"RAG": ["search(original)", "query", "summary", "one-to-many query"]},
        {"data": ["eda", "cleaning", "preprocessing", "feature construction", "feature selection", "modelling"]},
        {"live": ["wti", "watchlist", "heatmap"]}
    }
}
    
    
    "productivity": ,
    
    }







