from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np

from tiingo import TiingoClient

from src.io.fetch import validate_date
from config.config import ASSET_CONFIGS
from src.io.parquet import ParquetManager


class TiingoAPI:
    def __init__(self, sleep_s):
        #https://www.tiingo.com/documentation/end-of-day
        self._sleep_s = sleep_s

    def _fetch_history(self, api_key, symbol, frequency, startDate): 
        if startDate is not None and not validate_date(startDate):
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
        
        
class PriceManager: 
    def __init__(self, sources, history_start):
        self._basepath = Path.cwd() / "data" / "raw" / "historical_prices" 
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._history_start = history_start
        self._identifiers = ["symbol", "date", "time"]
        self._sleep_s = 0.2
        self._batch_size = 100
        self._config = ASSET_CONFIGS
        self._today = datetime.date.today()
        self._partition_degree = 2 #overhead (n1 x n2 x n3)
        
        self._writer = ParquetManager()
        self._reader = ParquetManager()
        self._sources = sources

        self._tiingo = TiingoAPI(0.5)

    def _load_batch(self, asset_type, frequency, batch_df, exclude_lazy, filedir, partition_map, refresh_threshold_days, REFRESH):
        assert batch_df.columns[0] == "symbol", "First column of each batch_df should be 'symbol'"
        lf, _ = self._reader.read_lazy(
            filedir.joinpath(*[f"{col}={val}" for col, val in partition_map.items()]),
            latest_by_identifiers=self._identifiers)
        
        batch_success = []
        batch_failed = []
        for row in batch_df.iter_rows(named=False):
            time.sleep(self._sleep_s)
            startDate = self._history_start
            symbol = row[0]      

            #SKIP -> Already Cached
            if not REFRESH:
                df_old = self._reader.filter_lazy(lf, filters={"symbol": [symbol]}, COLLECT=True) 
                exist_bool = not (df_old is None or df_old.is_empty())
                if exist_bool and "date" in df_old.columns:
                    if refresh_threshold_days is None:
                        continue
                    maxDate = df_old["date"].max()
                    startDate = maxDate.strftime("%Y-%m-%d") 
                    days_elapsed = (self._today - maxDate).days
                    if days_elapsed <= refresh_threshold_days:
                        continue
                
            #FETCH -> Symbol
            for source, api_key in self._sources.items():
                time.sleep(self._sleep_s)
                df_new = None
                
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
                    print(f"[INFO]Fetching {symbol}: {startDate}")
                    if source == "tiingo":                        
                        df_new = self._tiingo._fetch_history(api_key, symbol, frequency, startDate)
                    else:
                        raise ValueError(f"Unknown source '{source}'")
                    
                    if df_new is None or df_new.is_empty():
                        raise ValueError(f"'{source}': is empty")

                    df_new = df_new.with_columns(pl.lit(source).alias("source"))                                
                    batch_success.append(df_new)
                    print(f"[INFO]Successfully saved {symbol}")
                    break #only uses the first valid source
                except Exception as e:
                    print(f"[WARNING]Error fetching {symbol}: {e}")
                    df_new = {"symbol": symbol, "asset_type": asset_type, "frequency": frequency, "source": source, "error": str(e).split("\n")[0][:100]}
                    batch_failed.append(df_new)
                
        if batch_success:
            batch_success = pl.concat(batch_success)
            self._writer.save_parquet(filedir.joinpath(*[f"{col}={val}" for col, val in partition_map.items()]), batch_success, partition_cols=None)  #batch ~ reduce io ovehead from R/W opertions by holding in memory
        return batch_failed

    def load_history(self, asset_type, database, frequency, refresh_threshold_days=None, REFRESH=False):
        if database is None or database.is_empty():
            print("[WARNING]Invalid database")
            return None, None, None
        assert isinstance(database, pl.DataFrame), "Should be polars dataframe"
        if asset_type not in self._config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        assert set(self._config[asset_type]["columns"]).issubset(database.columns), "DataFrame should be a database from DM"
        partition_cols = self._config[asset_type]["partitions"][:self._partition_degree] #frequently filtered + low cardinality
        database = database.with_columns([pl.col(c).cast(pl.Utf8).fill_null("Unknown") for c in partition_cols])
        exclude_dir = self._basepath / "exclude" 
        exclude_lazy, _ = self._reader.read_lazy(exclude_dir, latest_by_identifiers=["symbol"])
    
        database = database.select(["symbol"]+[c for c in database.columns if c != "symbol"])
        filedir = self._basepath / asset_type / frequency
        params = {"asset_type": asset_type, "frequency": frequency, "filedir": filedir, \
                  "exclude_lazy": exclude_lazy, "refresh_threshold_days": refresh_threshold_days, "REFRESH": REFRESH}

        batch_idx = 0
        total_calls = len(database)
        failed_calls = []
        print(f"\n[INFO]Fetching Historical Prices ({total_calls})")
        assert set(partition_cols).issubset(database.columns), "Partition cols should be present in the DataFrame"
        subtables = database.partition_by(partition_cols, as_dict=True)
        for keys, subdf in subtables.items():
            partition_map = {col: val for col, val in zip(partition_cols, keys)}
            for offset in range(0, subdf.height, self._batch_size):
                print(f"[INFO]Batch {batch_idx}")
                batch_idx +=1
                batch_df = subdf.slice(offset, self._batch_size)
                batch_failed = self._load_batch(**params, batch_df=batch_df, partition_map=partition_map) 
                failed_calls.extend(batch_failed)
        del batch_df, batch_failed
        
        success_rate = ((total_calls - len(failed_calls)) / total_calls) * 100 if total_calls>0 else np.nan
        print(f"[INFO]Successfully fetched: {success_rate:.2f}% of symbols ({total_calls})")

        symbols = list(database["symbol"].unique())
        lf, partition_cols = self._reader.read_lazy(filedir, latest_by_identifiers=self._identifiers) 
        output_df = self._reader.filter_lazy(lf, filters={"symbol": symbols}, COLLECT=True)
        
        failed_df = pl.DataFrame(failed_calls) if failed_calls else None
        self._writer.save_parquet(exclude_dir, failed_df, partition_cols=None) 
        return output_df, failed_df

    def compact_job(self, asset_type, frequency): 
        filedir = self._basepath / asset_type / frequency #assumes same schema
        self._writer.compact_job(filedir)
        return self



    
