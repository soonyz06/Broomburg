from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np

from tiingo import TiingoClient

from config.config import ASSET_CONFIGS
from src.io.fetch import validate_date
from src.io.parquet import ParquetManager


class PriceManager: 
    def __init__(self, pm: ParquetManager, sources):
        self._config = ASSET_CONFIGS
        self._basepath = Path.cwd() / "data" / "raw" / "historical_prices" 
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._history_start = "2026-01-01" #dry run
        self._today = datetime.date.today()
        self._identifiers = ["symbol", "date", "time"]
        self._partition_degree = 2 #overhead (n1 x n2 x n3)

        self._writer = pm
        self._reader = pm
        self._sources = sources 

    def _fetch_history_tiingo(self, api_key, symbol, frequency, startDate): 
        print(f"[INFO]Fetching {symbol}: {startDate}")
        if not validate_date(startDate):
            return None

        df_new = TiingoClient({'api_key': api_key}).get_dataframe(
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
        if asset_type not in self._config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        if database is None or database.is_empty():
            print("[WARNING]Invalid database")
            return None, None

        partition_cols = self._config[asset_type]["partitions"][:self._partition_degree] #frequently filtered + low cardinality
        database = database.with_columns([pl.col(c).cast(pl.Utf8).fill_null("Unknown") for c in partition_cols])
        filedir = self._basepath / "exclude" 
        exclude_lazy = self._reader.read_lazy(filedir, identifiers=["symbol"])

        total_calls = len(database)
        failed_calls = []
        max_days_elapsed = 0
        print(f"\n[INFO]Fetching Historical Prices ({total_calls})")
        for row in database.iter_rows(named=True): 
            startDate = self._history_start
            symbol = row["symbol"]
            filedir = self._basepath / asset_type / frequency
            filedir = filedir.joinpath(*[f"{c}={row[c]}" for c in partition_cols]) #partition to treat as one file (same schema) 
            lf = self._reader.read_lazy(filedir, identifiers=self._identifiers)

            #SKIP -> Already Cached
            if not REFRESH: #REFRESH currently reloads entire dataset
                df_old = self._reader.filter_lazy(lf, filters={"symbol": [symbol]}, COLLECT=True)
                exist_bool = not (df_old is None or df_old.is_empty())
                if exist_bool and "date" in df_old.columns:
                    startDate = df_old["date"].max().strftime("%Y-%m-%d") 
                    days_elapsed = (self._today - df_old["date"].max()).days
                    if (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
                        max_days_elapsed = max(days_elapsed, max_days_elapsed)
                        continue
                
            #FETCH -> Symbol
            for source, api_key in self._sources.items(): 
                #Skip -> Already Failed
                if exclude_lazy is not None: #filtered at source level, stored at asset_type level
                    IsInExclude = not self._reader.filter_lazy(
                        exclude_lazy,
                        filters={"symbol": [symbol], "asset_type": [asset_type], "frequency": [frequency], "source": [source]}, inclusive=True
                        ).limit(1).collect().is_empty()
                    if IsInExclude:
                        continue 

                #FETCH -> Source
                try:
                    if source == "tiingo":                        
                        df_new = self._fetch_history_tiingo(api_key, symbol, frequency, startDate)
                    else:
                        raise ValueError(f"Unknown source '{source}'")
                    if df_new is None:
                        raise ValueError(f"Empty source '{source}'")
                    
                    self._writer.append_parquet(filedir, df_new, source=source, identifiers=self._identifiers)
                    print(f"[INFO]Successfully saved {symbol}")
                    break #any source would work
                except Exception as e:
                    print(f"[WARNING]Error fetching {symbol}: {e}")
                    df_new = {"symbol": symbol, "asset_type": asset_type, "frequency": frequency, "source": source, "error": str(e).split("\n")[0][:100]}
                    failed_calls.append(df_new)
                time.sleep(1.2)
                
        success_rate = ((total_calls - len(failed_calls)) / total_calls) * 100 if total_calls>0 else np.nan
        print(f"[INFO]Successfully fetched: {success_rate:.2f}% of symbols ({total_calls})")
        print(f"[INFO]Maximum days elapsed: {max_days_elapsed}")

        symbols = list(database["symbol"].unique())
        filedir = self._basepath / asset_type / frequency
        lf = self._reader.read_lazy(filedir, identifiers=self._identifiers) 
        output_df = self._reader.filter_lazy(lf, filters={"symbol": symbols}, COLLECT=True)
        
        failed_df = pl.DataFrame(failed_calls) if failed_calls else None
        filedir = self._basepath / "exclude" 
        self._writer.append_parquet(filedir, failed_df, source=None, identifiers=["symbol"]) 
        return output_df, failed_df
