import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.manager import ManagerManager
from src.io.parquet import ParquetManager
from src.io.database import DatabaseManager
from src.io.price import PriceManager
from src.io.sec import SECManager


t0 = log_info("Fetch")
api_keys = load_apikeys()
pm = ParquetManager()
mm = ManagerManager(pm)
history_start = "2026-01-01"

asset_type = "equities" 
filters = {
    "currency": ['USD'],
    "exchange": ['NMS', 'NYQ', 'NGM', 'NCM', 'ASE', 'PCX', 'PNK'],
    "market_cap": ["Mega Cap"]
}
limit = 3 #dry run


dm = DatabaseManager(pm=pm)
df, exclude = dm.filter_database(asset_type, filters=filters, limit=limit, REFRESH=False)
log_df(df, "DM loaded")
log_df(exclude, "DM excludes")
print("-"*120)

sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
pm = PriceManager(mm=mm, sources=sources, history_start=history_start)
df, partition_cols, failed_df = pm.load_history(asset_type, df, REFRESH=False)
log_df(df, "PM loaded")
log_df(failed_df, "PM excludes")
print(f"Partitioned by: {partition_cols}")
print("-"*120)

sources = {"edgar": None}
sm = SECManager(mm=mm, sources=sources, history_start=history_start)
output_map, partition_cols, failed_df = sm.load_sec(df, form="10-K", funcs=["statements", "sections"], REFRESH=False)
for func, df in output_map.items():
    log_df(df, f"SM {func} loaded")
log_df(failed_df, "SM excludes")
print(f"Partitioned by: {partition_cols}") ##smth not right tf, why url in col
print("-"*120)

##Check: Compact job, markdowns, ifrs(nvo, pdd), csv check 
log_info("Fetch", t0)


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 
#Coupling, Locality

#On demand vs Cached: ?(ml | offline)
RAW={
    "macro": ["fred", "oecd", "wb"],
    "news": ["rss", "guardian", "8k", "s-1"],
    "analyst": ["ratings", "earnings", "estimates"],
    "alt": ["google trends", "3/4/5", "13F/D/G"],
    "high-level": ["databento"]
     }         

APP = {
    "nlp": {
        "rag": ["query+keyword search", "web search", "cache search", "dynamic search"], #context learning
        "llm": ["source", "reasoning", "query decomposition", "entity decomposition"], #treat as human, recreate my workflow (warren)
        "tool": ["call functions", "sentiment"]
    },
    "ml": {
        "data": ["eda", "cleaning", "preprocessing", "feature construction", "feature selection", "modelling"]
    },
    "productivity": { #llm to emulate workflow
        "notes": ["adjacency list with set", "free-moving network graph with forces (center, repel and link)"],
        "todo": ["priority", "calender", "recurring", "completed"],
        "clock": ["timer", "alarm"],
        "storage": ["upload", "download", "query"],
        "inbox": [],
        "bookmark": []
    },
    "financial": {
        "cached": ["display", "export", "import"],
        "live": ["wti", "watchlist", "heatmap", "chart"]
    }
}
    






