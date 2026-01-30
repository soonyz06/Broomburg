import polars as pl
import pandas as pd
from pathlib import Path
import time
import datetime
import os
import numpy as np

from tiingo import TiingoClient
from config.config import ASSET_CONFIGS

from src.io.write import Writer
from src.io.fetch import validate_date


class ManagerUtils: 
    @staticmethod 
    def read_lazy(filedir, identifiers, filename=None): 
        filedir.mkdir(parents=True, exist_ok=True)
        if filename is None:        
            files = list(filedir.rglob(f"chunk_*.parquet"))
        else:
            files = list(filedir.rglob(f"*/{filename}/chunk_*.parquet"))
        if not files:
            return None
        lf = pl.scan_parquet(files)
        if identifiers is None:
            return lf
            
        latest = lf.group_by(identifiers).agg(pl.col("timestamp").max().alias("timestamp")) #lazy filtering: deduplicate at read
        lf = latest.join(lf, on=identifiers+["timestamp"], how="inner") #faster than global sort + unique, but uses all latest rows per group instead of only one
        return lf

    @staticmethod
    def filter_lazy(lf, filters=None, inclusive=True, COLLECT=False): #in isolation, aggeragate -> join (semi, anti)
        if lf is None:
            return None
        
        filters = filters or {}
        schema_names = lf.collect_schema().names()
        for key, val in filters.items():
            if key not in schema_names:   
                print(f"[WARNING]{key} is an invalid filter key")
                continue
            if val is None:
                continue
            val = val if isinstance(val, list) else [val]
            lf = lf.filter(pl.col(key).is_in(val)) if inclusive else lf.filter(pl.col(key).is_in(val).not_())            
        return lf.collect() if COLLECT else lf

    @staticmethod
    def clear_folder(filedir):
        filedir.mkdir(parents=True, exist_ok=True)
        files = list(filedir.rglob(f"chunk_*.parquet"))
        for f in files:
            if f.exists():
                print(f"Deleting {f}") #dry run
                #f.unlink()
        return

    @staticmethod
    def compact_job(self, filedir, identifiers=["symbol"], output_name="compacted.parquet"): #merging and deduplication 
        lf = ManagerUtils.read_lazy(filedir, identifiers)
        #[Make Changes Here]
        #ManagerUtils.clear_folder(filedir) #override
        writer.append_parquet(filedir, df, source=None, identifiers, FORCENEW=True) #(file-level versioning)
        return 
    
    @staticmethod   
    def sample_df(df, sampling_rate=None, seed=42):
        if df is None:
            return None 
        assert isinstance(database, pl.DataFrame), "Should be polars dataframe"
        
        if sampling_rate is None or sampling_rate < 0:
            return df
        elif sampling_rate <= 1:
            return df.sample(fraction=sampling_rate, seed=seed)
        else:
            return df.sample(n=min(sampling_rate, df.height), seed=seed)
        

class DatabaseManager:
    def __init__(self, writer: Writer):
        self.config = ASSET_CONFIGS
        self.basepath = Path.cwd() / "data" / "raw" / "database"
        self.basepath.mkdir(parents=True, exist_ok=True)
        self.identifiers = ["symbol"] #const
        self.size_map = {'Mega Cap': 0, 'Large Cap': 1, 'Mid Cap': 2, 'Small Cap': 3, 'Micro Cap': 4, 'Nano Cap': 5, None: 6}

        self.writer = writer
        
    def fetch_database(self, asset_type, REFRESH=False, COLLECT=False): #initial only, use managerutils after
        if asset_type not in self.config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        config = self.config[asset_type]
        
        filedir = self.basepath / f"{asset_type}" / "table"
        if not REFRESH:
            lf = ManagerUtils.read_lazy(filedir, identifiers=self.identifiers)
            if lf is not None:
                lf = lf.select(config["columns"])
                return lf.collect() if COLLECT else lf
        
        df = pl.from_pandas(config["fetcher"]().reset_index())
        df_final = df.select(config["columns"])
        self.writer.append_parquet(filedir, df_final, source="fd", identifiers=self.identifiers)
        return df_final if COLLECT else df_final.lazy()

    def filter_database(self, asset_type, filters={}, limit=None, REFRESH=False, COLLECT=True):
        lf = self.fetch_database(asset_type, REFRESH=REFRESH, COLLECT=False)
        lf = ManagerUtils.filter_lazy(lf, filters=filters, COLLECT=False)

        exclude_lazy = ManagerUtils.read_lazy(self.basepath.parent, filename="exclude", identifiers=self.identifiers)
        exclude_lazy = ManagerUtils.filter_lazy(exclude_lazy, filters={"asset_type": asset_type, "error": None}, inclusive=True) #union of all excludes (by error)
        if exclude_lazy is None:
            lf = self.n_lazy(lf, asset_type, limit, COLLECT=COLLECT)
            return lf, None
        lf = lf.join(exclude_lazy.select(self.identifiers), on="symbol", how="anti")
        lf = self.n_lazy(lf, asset_type, limit, COLLECT=COLLECT)
        exclude_lazy = exclude_lazy.limit(limit)
        return lf, exclude_lazy.collect() if COLLECT else exclude_lazy
    
    def n_lazy(self, lf, asset_type, limit, COLLECT=True):
        if asset_type == "equities":
            lf = lf.with_columns(pl.col("market_cap").replace(self.size_map).alias("size")).sort(["size", "symbol"], nulls_last=True).drop("size")
        lf = lf.limit(limit)
        return lf.collect() if COLLECT else lf

        
class PriceManager: ##abstract for all types of data
    def __init__(self, writer: Writer, api_key):
        self.config = ASSET_CONFIGS
        self.basepath = Path.cwd() / "data" / "raw" / "historical_prices" 
        self.basepath.mkdir(parents=True, exist_ok=True)
        self.history_start = "2026-01-01" #dry run
        self.today = datetime.date.today()
        self.identifiers = ["symbol", "date", "time"]
        self.sources = ["tiingo"]
        self.partition_degree = 2 #overhead (n1 x n2 x n3)

        self.writer = writer
        self.client = TiingoClient({'api_key': api_key})

    def fetch_history_tiingo(self, symbol, frequency, startDate=None): 
        print(f"[INFO]Fetching {symbol}: {startDate}")
        if startDate is None:
            startDate = self.history_start
        validate_date(startDate)

        df_new = self.client.get_dataframe(
            symbol,
            frequency=frequency,
            startDate=startDate
        )
        df_new = pl.from_pandas(df_new.reset_index())
        df_new = df_new.with_columns([
            pl.col("date").dt.time().alias("time"),
            pl.col("date").dt.date().alias("date"),
            pl.lit(symbol).alias("symbol"),
        ])
        return df_new

    def load_history(self, asset_type, database, frequency="daily", refresh_threshold_days=None, REFRESH=False): 
        assert isinstance(database, pl.DataFrame), "Should be polars dataframe"
        if asset_type not in self.config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        if database is None or database.is_empty():
            print("[WARNING]Invalid database")
            return None, None

        partition_cols = self.config[asset_type]["partitions"][:self.partition_degree] #frequently filtered + low cardinality
        database = database.with_columns([pl.col(c).cast(pl.Utf8).fill_null("Unknown") for c in partition_cols])
        filedir = self.basepath / "exclude" 
        exclude_lazy = ManagerUtils.read_lazy(filedir, identifiers=["symbol"])
        if exclude_lazy is not None:
            x = exclude_lazy.collect()

        total_calls = len(database)
        failed_calls = []
        max_days_elapsed = 0
        print(f"\n[INFO]Fetching Historical Prices ({total_calls})")
        for row in database.iter_rows(named=True): 
            startDate = None
            symbol = row["symbol"]
            filedir = self.basepath.joinpath(f"{frequency}", f"{asset_type}", *[f"{c}={row[c]}" for c in partition_cols]) #partition to treat as one file (same schema) 
            lf = ManagerUtils.read_lazy(filedir, identifiers=self.identifiers)

            #SKIP -> Already Cached
            if not REFRESH: #REFRESH currently reloads entire dataset
                df_old = ManagerUtils.filter_lazy(lf, filters={"symbol": [symbol]}, COLLECT=True)
                exist_bool = not (df_old is None or df_old.is_empty())
                if exist_bool and "date" in df_old.columns:
                    startDate = df_old["date"].max().strftime("%Y-%m-%d") 
                    days_elapsed = (self.today - df_old["date"].max()).days
                    if (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
                        max_days_elapsed = max(days_elapsed, max_days_elapsed)
                        continue
                
            #FETCH -> Symbol
            for source in self.sources: #or use dict to map each freq/asset_type to their sources
                #Skip -> Already Failed
                if exclude_lazy is not None: #filtered at source level, stored at asset_type level
                    IsInExclude = not ManagerUtils.filter_lazy(
                        exclude_lazy,
                        filters={"symbol": [symbol], "freq": [frequency], "asset_type": [asset_type], "source": [source]}, inclusive=True
                        ).limit(1).collect().is_empty()
                    if IsInExclude:
                        continue 

                #FETCH -> Source
                try:
                    if source == "tiingo":                        
                        df_new = self.fetch_history_tiingo(symbol, frequency, startDate)
                    else:
                        raise ValueError(f"Unknown source '{source}'")
                    self.writer.append_parquet(filedir, df_new, source=source, identifiers=self.identifiers)
                    print(f"[INFO]Successfully saved {symbol}")
                    break
                except Exception as e:
                    print(f"[WARNING]Error fetching {symbol}: {e}")
                    df_new = {"symbol": symbol, "freq": frequency, "asset_type": asset_type, "source": source, "error": str(e).split("\n")[0][:100]}
                    failed_calls.append(df_new)
                time.sleep(1.2)
                
        success_rate = ((total_calls - len(failed_calls)) / total_calls) * 100 if total_calls>0 else np.nan
        print(f"[INFO]Successfully fetched: {success_rate:.2f}% of symbols ({total_calls})")
        print(f"[INFO]Maximum days elapsed: {max_days_elapsed}")

        symbols = list(database["symbol"].unique())
        filedir = self.basepath / f"{frequency}" / f"{asset_type}" 
        lf = ManagerUtils.read_lazy(filedir, identifiers=self.identifiers) 
        output_df = ManagerUtils.filter_lazy(lf, filters={"symbol": symbols}, COLLECT=True)
        
        failed_df = pl.DataFrame(failed_calls) if failed_calls else None
        filedir = self.basepath / "exclude" 
        self.writer.append_parquet(filedir, failed_df, source=None, identifiers=["symbol"]) 
        return output_df, failed_df
    
##raw: edgar, fred, oecd, databento
        








        
