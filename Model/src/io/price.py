from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta

from tiingo import TiingoClient

from src.io.fetch import validate_date
from src.io.manager import ManagerManager

class FXManager():
    def __init__(self):
        self._basepath = Path.cwd() / "data" / "raw" / "FX"
        self._basepath.mkdir(parents=True, exist_ok=True)
        
    def _fetch_fx(self, symbol, start, end):
        print(f"Fetching exchange rate ({symbol})")
        fx = Ticker(symbol)
        start = start - relativedelta(months=1)
        end   = end + relativedelta(months=1)
        df = pl.from_pandas(fx.history(start=start, end=end, interval="1d").reset_index()).select(["date", "close"])
        return df

    def _get_fx_data(self, currecy, start, end):
        if currecy == "USD":
            return None
        symbol = f"USD{currecy}=X"
        file_path =  self._basepath / f"{symbol}.parquet"
        
        try:
            df = pl.read_parquet(file_path)
            start_loaded = df["date"].min()
            end_loaded = df["date"].max()
        except FileNotFoundError:
            start_loaded=None
            end_loaded=None
            
        if start_loaded is None or end_loaded is None:
            df = self._fetch_fx(symbol, start, end)
        elif start<start_loaded or end>end_loaded:
            new_df = self._fetch_fx(symbol, start, end)
            df = pl.concat([df, new_df]).unique(subset=["date"], keep="last").sort("date")
        else:
            return df
        df.write_parquet(file_path)
        return df

    def adj_fx_table(self, df):
        fx_df = {}
        results = []
        currencies = df["currencyCode"].unique()
        for currency in currencies:
            fx_df[currency] = self._get_fx_data(currency, df["date"].min(), df["date"].max())
            temp_df = df.filter(pl.col("currencyCode")==currency).sort("date")
            if fx_df[currency] is not None:
                fx_data = fx_df[currency].with_columns(pl.col("date").cast(df["date"].dtype))
                temp_df = temp_df.join_asof(fx_data, on="date", strategy="backward").rename({"close": "fx"})
            else:
                temp_df = temp_df.with_columns(pl.lit(1, dtype=pl.Float64).alias("fx"))
            results.append(temp_df)
        df = pl.concat(results).sort("date")
        
        df = df.with_columns(
            (pl.col(col) / pl.col("fx")).cast(pl.Float64).alias(col)
            for col in df.select(pl.selectors.numeric()).columns
            if not any(term in col.lower() for term in ["shares", "issuance", "market"]) and col != "fx"
        )
        return df

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
    def __init__(self, mm: ManagerManager, sources, history_start):
        self._basepath = Path.cwd() / "data" / "raw" / "historical_prices" 
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._history_start = history_start
        self._identifiers = ["symbol", "date", "time"]
        self._sleep_s = 0.2
        self._batch_size = 100

        self._manager = mm
        self._writer = self._manager._writer
        self._reader = self._manager._reader
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
                days_elapsed, startDate = self._manager.get_days_elapsed(lf, filters={"symbol": [symbol]}, startDate=startDate)
                if days_elapsed is not None and (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
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
                    
                    if df_new is None:
                        raise ValueError(f"'{source}': {func} is empty")
                    elif isinstance(df_new, pl.DataFrame):
                        if df_new.is_empty():
                            raise ValueError(f"'{source}': {func} is empty")
                        else:
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
            return None, None, None
        partition_cols, database, exclude_lazy, exclude_dir = self._manager.initialise_manager(asset_type, database, self._basepath)
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
        return output_df, partition_cols, failed_df

    def compact_job(self, asset_type, frequency): 
        filedir = self._basepath / asset_type / frequency #assumes same schema
        self._writer.compact_job(filedir)
        return self



    
