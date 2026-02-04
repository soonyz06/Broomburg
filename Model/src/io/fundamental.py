from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np

from src.io.price import FXManager
from yahooquery import Ticker, Screener


class YahooQueryAPI:
    def __init__(self, sleep_s):
        self._sleep_s = sleep_s
        self._fx = FXManager()

    def _replace_metric(self, df, des, src):
        if des not in df.columns:
            df = df.with_columns(pl.col(src).alias(des))
        else:
            df = df.with_columns(
                pl.when(pl.col(des).is_null())
                  .then(pl.col(src))
                  .otherwise(pl.col(des))
                  .alias(des)
            )
        return df

    def _fetch_financial(self, symbols, domain, freq="annual"):
        time.sleep(self._sleep_s)
        ticker = Ticker(symbols, asynchronous=True)
        print(f"Fetching {domain.title()}({freq[0]}) -> {len(symbols)}")
            
        if domain == "income_statement":
            retrieved = ticker.income_statement(frequency=freq, trailing=False)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="NetIncome")
        elif domain == "cash_flow":
            retrieved = ticker.cash_flow(frequency=freq, trailing=False)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="OperatingCashFlow")
        elif domain == "balance_sheet":
            retrieved = ticker.balance_sheet(frequency=freq)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="CommonStockEquity")
            df = df.with_columns(pl.col("OrdinarySharesNumber").forward_fill().over("symbol"))
        else:
            print("Invalid domain")
            return None
        
        df = df.with_columns(pl.col("asOfDate").dt.date().alias("date")).drop("asOfDate")
        df = df.select(["date"] + [c for c in df.columns if c != "date"])
        return df if not df.is_empty() else None

    def get_combined_financials(self, symbols, freq): ##get list of all missing, fetch async, dete4rminstic schema :)
        domains = ["income_statement", "cash_flow", "balance_sheet"]
        identifiers = ["symbol", "date", "currencyCode"]
        for i, domain in enumerate(domains):
            df = self._fetch_financial(symbols, domain, freq)
            if not isinstance(df, pl.DataFrame):
                return None
        
            if i==0:
                combined_df = df
            else:
                combined_df = combined_df.join(df, on=identifiers, how="inner")
                combined_df = combined_df.drop([c for c in combined_df.columns if c.endswith("_right")])
        df = combined_df.select(identifiers+[c for c in combined_df.columns if c not in identifiers])
        df = self._replace_metric(df, "DilutedAverageShares", "BasicAverageShares")
        df = self._replace_metric(df, "DilutedAverageShares", "OrdinarySharesNumber")
        df = self._replace_metric(df, "CashFlowFromContinuingOperatingActivities", "OperatingCashFlow")
        df = self._replace_metric(df, "CashFlowFromContinuingInvestingActivities", "InvestingCashFlow")
        df = self._replace_metric(df, "CashFlowFromContinuingFinancingActivities", "FinancingCashFlow")
        df = self._replace_metric(df, "OperatingCashFlow", "CashFlowFromContinuingOperatingActivities")
        df = self._replace_metric(df, "CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents")
        df = self._fx.adj_fx_table(df)
        return df

    def _fetch_history(self, symbols, period, interval):
        time.sleep(self._sleep_s)
        print(f"Fetching History({period},{interval}) -> {len(symbols)}")
        ticker = Ticker(symbols)
        retrieved = ticker.history(period=period, interval=interval)
        df = pl.from_pandas(retrieved.reset_index()).select(["symbol", "date", "open", "high", "low", "close", "volume", "adjclose"])
        df = df.with_columns([pl.col(c).round(2) for c in df.columns if c not in ["symbol", "date"]])
        return df if not df.is_empty() else None


class FundamentalManager: 
    def __init__(self, mm: ManagerManager, sources, history_start):
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

        self._yq = YahooQueryAPI(0.5)

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
                if exclude_lazy is not None: 
                    IsInExclude = not self._reader.filter_lazy(
                        exclude_lazy,
                        filters={"symbol": [symbol], "asset_type": [asset_type], "frequency": [frequency], "source": [source]}, inclusive=True
                        ).limit(1).collect().is_empty()
                    if IsInExclude:
                        continue 

                #FETCH -> Source
                try:
                    print(f"[INFO]Fetching {symbol}: {startDate}")
                    if source == "yahooquery":                        
                        df_new = self._yq.get_combined_financials(symbol, frequency)
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
                    break 
                except Exception as e:
                    print(f"[WARNING]Error fetching {symbol}: {e}")
                    df_new = {"symbol": symbol, "asset_type": asset_type, "frequency": frequency, "source": source, "error": str(e).split("\n")[0][:100]}
                    batch_failed.append(df_new)
                
        if batch_success:
            batch_success = pl.concat(batch_success)
            self._writer.save_parquet(filedir.joinpath(*[f"{col}={val}" for col, val in partition_map.items()]), batch_success, partition_cols=None)  
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
        return output_df, partition_cols, failed_df

    def compact_job(self, asset_type, frequency): 
        filedir = self._basepath / asset_type / frequency 
        self._writer.compact_job(filedir)
        return self


