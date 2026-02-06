from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np
import json
from dateutil.relativedelta import relativedelta

from src.utils.df import replace_metric
from yahooquery import Ticker


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
        if df is None or df.is_empty():
            return None
        
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
    
class FAProcesser:
    def _get_expression(self, df, x, op, dtype, key):
        missing = [col for col in x if (col not in df.columns) and isinstance(col, str)]
        x_filled = [(pl.col(name).fill_null(0) if name in df.columns else pl.lit(0)) for name in x]

        if op == "sum":
            expr = sum(x_filled)
        elif op == "diff":
            expr = x_filled[0] - sum(x_filled[1:])
            
        elif missing:
            expr = pl.lit(None)
        elif op=="":
            expr = pl.col(x[0])
        elif op == "proportion":
            expr = (pl.col(x[0]) * x[1]).round(6)
        elif op == "log":
            expr = pl.when(pl.col(x[0])>0).then(pl.col(x[0]).log1p()).otherwise(pl.lit(0)).round(6)
        elif op == "mult":
            expr = (pl.col(x[0]) * pl.col(x[1])).round(6)
        elif op == "bool":
            expr = (pl.col(x[0]).fill_null(0) > 0)
        elif op == "max":
            expr = pl.col(x[0]).clip(x[1], None)
        elif op == "min":
            expr = pl.col(x[0]).clip(None, x[1])
            
        elif op == "divide":
            num, denom = pl.col(x[0]), pl.col(x[1])
            expr = (
                pl
                .when((num<=0) & (denom<=0))
                .then(pl.lit(None))
                .when(denom ==0)
                .then(pl.lit(None))
                .otherwise((num / denom).round(6))
            )
        elif op == "change":
            num, denom = pl.col(x[0]), pl.col(x[0]).shift(1).over("symbol")
            expr = (
                pl
                .when(((num > 0) & (denom <= 0)) | ((num <= 0) & (denom > 0)))
                .then(pl.lit(None))
                .when(denom ==0)
                .then(pl.lit(None))
                .otherwise(((num / denom - 1)).round(6))
            )
        else:
            expr = pl.lit(None)

        if dtype == pl.Int64: expr = expr.cast(pl.Float64).round(0)
        return expr.cast(dtype).alias(key)

    def add_metrics(self, df, metrics, NEW=False): 
        try:
            keys = ["symbol", "date"]
            types = {"float": pl.Float64, "int": pl.Int64, "bool": pl.Int8}
            
            for name, session_dict in metrics.items(): #{Section: {m, ..., m}}
                exprs = []
                if "hidden" not in name:
                    keys.extend(list(session_dict.keys()))
                for key, val in session_dict.items():  # m = New_Metric: [[Inputs], Operation, Data_Type]
                    inputs, operation, data_type = val[0], val[1], types[val[2]]
                    expr = self._get_expression(df, inputs, operation, data_type, key)
                    exprs.append(expr)
                df = df.with_columns(exprs) # Apply each section in one vectorized call
        except FileNotFoundError:
            print(f"[WARNING]{file_name}.json not found")
        return df if not NEW else df[keys]

    def _fetch_history(self, symbols, period, interval):
        time.sleep(self._sleep_s)
        print(f"Fetching History({period},{interval}) -> {len(symbols)}")
        ticker = Ticker(symbols)
        retrieved = ticker.history(period=period, interval=interval)
        df = pl.from_pandas(retrieved.reset_index()).select(["symbol", "date", "open", "high", "low", "close", "volume", "adjclose"])
        df = df.with_columns([pl.col(c).round(2) for c in df.columns if c not in ["symbol", "date"]])
        return df if not df.is_empty() else None

    
class YahooQueryAPI:
    def __init__(self, sleep_s, SIZE=10):
        self._sleep_s = sleep_s
        self._metrics, self.standard_cols = self._load_configs()
        self._SIZE = SIZE

        self._fx = FXManager()
        self._pro = FAProcesser()

    def _load_configs(self):
        basepath = file_path = Path.cwd() / "config" / "fa_config"
        basepath.mkdir(parents=True, exist_ok=True)

        metrics = {}
        standard_cols = ["date", "symbol"]
        for domain in ["income_statement", "cash_flow", "balance_sheet"]:
            filepath = basepath / f"{domain}_add.json"
            with open(filepath, "r") as f:
                metrics[domain] = json.load(f)
            for name, session_dict in metrics[domain].items():
                if "hidden" not in name:
                    standard_cols.extend(session_dict.keys())
        standard_cols = list(dict.fromkeys(standard_cols)) #deduplicate, maintaining order
        return metrics, standard_cols

    def _fetch_financial(self, symbols, domain, frequency="annual"):
        time.sleep(self._sleep_s)
        ticker = Ticker(symbols, asynchronous=True)
        print(f"Fetching {domain.title()}({frequency[0]}) -> {len(symbols)}")
            
        if domain == "income_statement":
            retrieved = ticker.income_statement(frequency=frequency, trailing=False)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="NetIncome")
        elif domain == "cash_flow":
            retrieved = ticker.cash_flow(frequency=frequency, trailing=False)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="OperatingCashFlow")
        elif domain == "balance_sheet":
            retrieved = ticker.balance_sheet(frequency=frequency)
            df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="CommonStockEquity")
            df = df.with_columns(pl.col("OrdinarySharesNumber").forward_fill().over("symbol"))
        else:
            print("Invalid domain")
            return None
        
        df = df.with_columns(pl.col("asOfDate").dt.date().alias("date")).drop("asOfDate")
        df = df.select(["date"] + [c for c in df.columns if c != "date"])
        return df if not df.is_empty() else None

    def get_combined_financials(self, symbols, frequency): 
        domains = ["income_statement", "cash_flow", "balance_sheet"]
        identifiers = ["symbol", "date", "currencyCode"]
        for i, domain in enumerate(domains):
            df = self._fetch_financial(symbols, domain, frequency)
            if not isinstance(df, pl.DataFrame):
                return None
        
            if i==0:
                combined_df = df
            else:
                combined_df = combined_df.join(df, on=identifiers, how="inner")
                combined_df = combined_df.drop([c for c in combined_df.columns if c.endswith("_right")])
        df = combined_df.select(identifiers+[c for c in combined_df.columns if c not in identifiers])
        df = replace_metric(df, "DilutedAverageShares", "BasicAverageShares")
        df = replace_metric(df, "DilutedAverageShares", "OrdinarySharesNumber")
        df = replace_metric(df, "CashFlowFromContinuingOperatingActivities", "OperatingCashFlow")
        df = replace_metric(df, "CashFlowFromContinuingInvestingActivities", "InvestingCashFlow")
        df = replace_metric(df, "CashFlowFromContinuingFinancingActivities", "FinancingCashFlow")
        df = replace_metric(df, "OperatingCashFlow", "CashFlowFromContinuingOperatingActivities")
        df = replace_metric(df, "CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents")

        df = self._fx.adj_fx_table(df)
        if df is None or df.is_empty():
            return None

        for domain in domains:
            df = self._pro.add_metrics(df, self._metrics[domain], NEW=False)
        df = df.with_columns([pl.lit(0).alias(c) for c in self.standard_cols if c not in df.columns])
        df = df.select(self.standard_cols) #determinstic schema
        return df.sort(["symbol", "date"])

    def fetch_batch(self, missing_symbols, function, params=None):
        if not missing_symbols:
            return None, []
        if params is None:
            params = {}

        all_frames = []
        failed_symbols = {}
        def process(batch, idx): #recursively
            while batch:
                try:
                    df = function(batch, **params)
                    if df is not None:
                        all_frames.append(df)
                        print(df.head(3))
                    return
                except Exception as e:
                    print(f"[WARNING] Batch {idx} failed: {batch} due to {e}")
                    if len(batch) == 1:
                        failed_symbols[batch[0]] = e
                        print(f"[ERROR] Single symbol failed: {batch[0]}")
                        return
                    mid = len(batch) // 2
                    process(batch[:mid], f"{idx}a")
                    process(batch[mid:], f"{idx}b")
                    return

        batches = [missing_symbols[i:i+self._SIZE] for i in range(0, len(missing_symbols), self._SIZE)]
        for i, batch in enumerate(batches):
            time.sleep(int(len(batch) * 0.5))
            process(batch, i)
            
        df = pl.concat(all_frames) if all_frames else None
        return df, failed_symbols


"""
rv = add_metrics(fa, "relative_valuation_add", NEW=True)

def get_tables(domains, fa, rv, exp):
    divisor = 10 ** exp
    dfs = {}
    costs = ["COGS", "SG&A", "R&D", "Others", "Operating Expense", "Taxes", "After-Tax Adj"]
    rename_maps = {}
    cols_maps = {}
    for domain in domains:
        file_path = Path.cwd() / "data" / "fa_config" / f"{domain}_rename.json"
        with open(file_path, "r") as f:
            rename_maps[domain] = json.load(f)
            
        file_path = Path.cwd() / "data" / "fa_config" / f"{domain}_add.json"
        with open(file_path, "r") as f:
            metrics = json.load(f)

        cols = ["date"]
        for name, session_dict in metrics.items():
            if "hidden" not in name:
                cols.extend(session_dict.keys())
        cols_maps[domain] = cols
        
    symbols = fa["symbol"].unique(maintain_order=True).to_list()
    for symbol in symbols:
        symbol_fa = fa.filter(pl.col("symbol") == symbol)
        data = {}

        for domain, line_items in rename_maps.items():
            cols = cols_maps[domain]
            temp_fa = symbol_fa.with_columns([pl.lit(None).alias(c) for c in cols if c not in symbol_fa.columns])
            temp_fa = temp_fa.select(cols).rename(line_items)
            temp_fa = temp_fa.with_columns((pl.col(pl.Int64) / divisor).round(0).cast(pl.Int64)) 
            temp_fa = temp_fa.with_columns((pl.col(c)*-1).alias(c) for c in costs if c in temp_fa.columns)
            data[domain] = temp_fa
        if rv is not None:
            data["relative_valuation"] = rv.filter(pl.col("symbol") == symbol).drop("symbol")
        dfs[symbol] = data
    return dfs

def format_subsets_df(df, subsets, highlights):
    if subsets is not None:
        for parent, children in subsets.items():
            for child in children:
                if child not in df.columns:
                    continue
                if parent in df.columns:
                    df = df.with_columns(pl.when(df[parent].is_null()).then(None).otherwise(df[child]).alias(child))
                if child in df.columns and df.select(pl.col(child).eq(0).all()).item(): #all zero
                    df = df.with_columns(pl.lit(None).alias(child))

    if highlights is not None:
        valid = [h for h in highlights if h in df.columns]
        df = df.select(["date"] + valid)

    df = df.with_columns([pl.when(pl.col(c) == -9999).then(None).otherwise(pl.col(c)).alias(c) for c in df.columns if df.schema[c] in (pl.Int64, pl.Float64)])
    mult_cols = [c for c in df.columns if c in ["EVE", "EVS", "EVB", "PE", "PS", "PB", "EPS", "Beta", "Close"]]
    float_cols = [c for c, dtype in df.schema.items()if c != "date" and dtype == pl.Float64 and c not in mult_cols]
    int_cols = [c for c, dtype in df.schema.items() if c != "date" and dtype == pl.Int64 and c not in mult_cols]
    df = df.with_columns(
        [pl.col(c).map_elements(lambda x: fmt(x, "m")).alias(c) for c in mult_cols] +
        [pl.col(c).map_elements(lambda x: fmt(x, "%")).alias(c) for c in float_cols] +
        [pl.col(c).map_elements(lambda x: fmt(x, ",")).alias(c) for c in int_cols]
    )
    df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns]).fill_null("")
    return df

df = format_subsets_df(df, subsets, highlights)
df = transpose_df(df, "date")
