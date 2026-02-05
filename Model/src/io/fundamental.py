from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np

from src.io.price import PriceManager
from src.io.yq import YahooQueryAPI


class FundamentalManager: 
    def __init__(self, mm: ManagerManager, sources, history_start, pm: PriceManager):
        self._basepath = Path.cwd() / "data" / "raw" / "fundamentals" 
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._history_start = history_start
        self._identifiers = ["symbol", "date"]
        self._sleep_s = 0.2
        self._batch_size = 100

        self._manager = mm
        self._writer = self._manager._writer
        self._reader = self._manager._reader
        self._sources = sources
        self._pm = pm

        self._yq = YahooQueryAPI(0.5, SIZE=10)

    def _load_batch(self, asset_type, frequency, batch_df, exclude_lazy, filedir, partition_map, refresh_threshold_days, REFRESH):
        assert batch_df.columns[0] == "symbol", "First column of each batch_df should be 'symbol'"
        lf, _ = self._reader.read_lazy(
            filedir.joinpath(*[f"{col}={val}" for col, val in partition_map.items()]),
            latest_by_identifiers=self._identifiers)
        
        batch_success = []
        batch_failed = []
        missing_rows = []
        for row in batch_df.iter_rows(named=False):
            time.sleep(self._sleep_s)
            startDate = self._history_start
            symbol = row[0]      

            #SKIP -> Already Cached
            if not REFRESH:
                days_elapsed, startDate = self._manager.get_days_elapsed(lf, filters={"symbol": [symbol]}, startDate=startDate)
                if days_elapsed is not None and (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
                    continue
                
            #FETCH -> Symbol
            for source, api_key in self._sources.items():
                time.sleep(self._sleep_s)
                df_new = None
                
                #Skip -> Already Failed
                if exclude_lazy is not None: 
                    IsInExclude = not self._reader.filter_lazy(
                        exclude_lazy,
                        filters={"symbol": [symbol], "asset_type": [asset_type], "frequency": [frequency], "source": [source]}, inclusive=True
                        ).limit(1).collect().is_empty()
                    if IsInExclude:
                        continue 

                #LOG -> To fetch
                missing_rows.append(row)
                break #1 source only

        if not missing_rows:
            return batch_failed

        #FETCH
        missing_database = pl.DataFrame(missing_rows, schema=batch_df.columns, orient="row")
        history, failed_df = self._pm.load_history(asset_type, missing_database, frequency="daily", refresh_threshold_days=7, REFRESH=False)
        history = history.select(self._identifiers+["close", "volume"])

        source = "yahooquery"
        missing_symbols = [row[0] for row in missing_rows]
        fundamentals, failed_symbols = self._yq.fetch_batch(missing_symbols, self._yq.get_combined_financials, params={"frequency": frequency}) #sorted and selected
        for symbol, error in failed_symbols.items():
            df_new = {"symbol": symbol, "asset_type": asset_type, "frequency": frequency, "source": source, "error": str(error).split("\n")[0][:100]}
            batch_failed.append(df_new)
            
        if fundamentals is None or fundamentals.is_empty():
            print(f"[WARNING]'{source}': is empty")
            return batch_failed
        
        fundamentals = fundamentals.with_columns(pl.lit(source).alias("source"))
        fundamentals = fundamentals.join_asof(history, on="date", by="symbol", strategy="backward", check_sortedness=False)
        fundamentals = fundamentals.with_columns((pl.col("close")*pl.col("DilutedAverageShares")).alias("MC"))
        self._writer.save_parquet(filedir.joinpath(*[f"{col}={val}" for col, val in partition_map.items()]), fundamentals, partition_cols=None)
        return batch_failed
        
    def load_fundamentals(self, asset_type, database, frequency, refresh_threshold_days=None, REFRESH=False):
        if database is None or database.is_empty():
            return None, None, None
        partition_cols, database, exclude_lazy, exclude_dir = self._manager.initialise_manager(asset_type, database, self._basepath)
        database = database.select(["symbol"]+[c for c in database.columns if c != "symbol"])
        filedir = self._basepath / asset_type / frequency
        params = {"asset_type": asset_type, "frequency": frequency, "filedir": filedir, \
                  "exclude_lazy": exclude_lazy, "refresh_threshold_days": refresh_threshold_days, "REFRESH": REFRESH}

        batch_idx = 0
        total_calls = len(database)
        failed_calls = []
        print(f"\n[INFO]Fetching Fundamentals ({total_calls})")
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
        filedir = self._basepath / asset_type / frequency 
        self._writer.compact_job(filedir)
        return self


