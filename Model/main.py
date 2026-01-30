import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, df_display
from src.io.fetch import load_apikeys
from src.io.managers import DatabaseManager, PriceManager, Writer


#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple


t0 = log_info("Fetch")
api_keys = load_apikeys()

asset_type = "equities" 
filters = {
    "currency": ['USD'],
    "exchange": ['NMS', 'NYQ', 'NGM', 'NCM', 'ASE', 'PCX', 'PNK'],
    "market_cap": ["Mega Cap"]
}
limit = 10 #dry run


    
wr = Writer()
dm = DatabaseManager(writer=wr)
df, exclude = dm.filter_database(asset_type, filters=filters, limit=limit, REFRESH=False)
#df_display(df, "DM loaded")
#df_display(exclude, "DM excludes")

pm = PriceManager(writer=wr, api_key=api_keys.get("TIINGO_API_KEY", None))
df, failed_df = pm.load_history(asset_type, df, REFRESH=False)
df_display(df, "PM loaded")


    
log_info("Fetch", t0)



#:)
##load data -> eda, cleaning, preprocessing, feature construction, feature selection, modelling
#plotly dash





