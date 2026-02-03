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

#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 
#On demand vs Cached: ?(ml | offline)


t0 = log_info("Fetch")
api_keys = load_apikeys()
pm = ParquetManager()
mm = ManagerManager(pm)

asset_type = "equities" 
filters = {
    "currency": ['usd'],
    "exchange": ['nms', 'nyq', 'ngm', 'ncm', 'ase', 'pcx', 'pnk'],
    "market_cap": ["mega_cap"]
}
limit = 20 #dry run
history_start = "2020-01-01"


dm = DatabaseManager(pm=pm)
lf = dm.fetch_database(asset_type, filters=filters, limit=None, REFRESH=False)
database, exclude = dm.filter_excludes(lf, asset_type, limit=limit, COLLECT=True)
log_df(database, "DM loaded")
log_df(exclude, "DM excludes")
print("-"*120)

sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None)}
frequency = "daily"
pm = PriceManager(mm=mm, sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
df, partition_cols, failed_df = pm.load_history(asset_type, database, frequency=frequency, REFRESH=False)
log_df(df, "PM loaded")
log_df(failed_df, "PM excludes")
print(f"Partitioned by: {partition_cols}")
print("-"*120)

form_key = "annual"
funcs = ["statements"]#, "sections"]
sources = {"edgar": None}
sm = SECManager(mm=mm, sources=sources, history_start=history_start)#.compact_job(asset_type, form_key=form_key, funcs=funcs)
output_map, partition_cols, failed_df = sm.load_sec(database, form_key=form_key, funcs=funcs, REFRESH=False)
if output_map is not None:
    for func, df in output_map.items():
        log_df(df, f"SM {func} loaded")
log_df(failed_df, "SM excludes")
print(f"Partitioned by: {partition_cols}") 
print("-"*120)



##nlp/rag, load sections on demand+cache, vector embedding, semantic chunking,, create vector lookup table
log_info("Fetch", t0) ##tqdm :) as outer update, using deterministic ssampling
    
##currency of irfs, fk it and use polygon?
APP = { #use by others
    "nlp": {
        "llm": ["source", "reasoning", "query decomposition", "entity decomposition", "loop"], #distill model, teach me qt
        "rag": ["upload", "query+keyword search", "web search", "cache search", "dynamic search", "store consolidated notes"], #context learning
        "actuators": ["call functions", "sentiment"],
        "orchestration": ["sequential", "hierarchical", "parallel"]
    },
    "ml": {
        "data": ["eda", "cleaning", "preprocessing", "feature construction", "feature selection", "modelling"]
    },
    "productivity": { #llm to emulate workflow (treat as human)
        "notes": ["targ+subtags", "adjacency list with set", "free-moving network graph with forces (center, repel and link)"],
        "todo": ["priority", "calender", "recurring", "completed"],
        "clock": ["timer", "alarm", "logger", "timespent"],
        "storage": ["upload", "download", "query"],
        "inbox": [],
        "bookmark": []
    },
    "financial": {
        "cached": ["display", "export", "import"],
        "live": ["wti", "watchlist", "heatmap", "chart"]
    }
}

RAW = {
    "macro": ["fred", "oecd", "wb"],
    "news": ["rss", "guardian", "8k", "s-1"],
    "analyst": ["ratings", "earnings", "estimates"],
    "alt": ["google trends", "3/4/5", "13F/D/G"],
    "high-level": ["databento"]
}     

#Abstraction, Coupling, Locality
#df check None and .is_empty()
#read: input latest_by_identifiers and output partition_cols
#save: input df requires partition_cols in df.columns and source, it adds partition_cols to filedir
#sleep: source, row

    






