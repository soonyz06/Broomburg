from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import json
import time
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta


class MetricProcessor:
    def _get_expression(self, col_set, x, op, dtype, key):
        missing = [col for col in x if isinstance(col, str) and col not in col_set]
        #missing = [col for col in x if (col not in df.columns) and isinstance(col, str)]
        x_filled = [
            (pl.col(name).fill_null(0) if (isinstance(name, str) and name in col_set) else pl.lit(name if not isinstance(name, str) else 0)) 
            for name in x
        ]
        #x_filled = [(pl.col(name).fill_null(0) if name in existing else pl.lit(0)) for name in x]

        if op == "sum":
            expr = pl.sum_horizontal(x_filled)
            #expr = sum(x_filled)
        elif op == "diff":
            expr = x_filled[0] - sum(x_filled[1:])
            
        elif missing and op!="change":
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

    def add_metrics(self, df, metrics): 
        types = {"float": pl.Float64, "int": pl.Int64, "bool": pl.Int8}
        
        for name, session_dict in metrics.items(): #{Section: {m, ..., m}}
            exprs = []
            col_set = set(df.columns)
            for key, val in session_dict.items():  # m = New_Metric: [[Inputs], Operation, Data_Type]
                inputs, operation, data_type = val[0], val[1], types[val[2]]
                expr = self._get_expression(col_set, inputs, operation, data_type, key)
                exprs.append(expr)
            df = df.with_columns(exprs) # Apply each section in one vectorized call        
        return df   

class CrossSectionManager:
    #description of time series up to a point in time 
    def __init__(self, target="returns"):
        self.feat = {
            "id": ["symbol", "date"],
            "num": [],
            "cat": [],
            "target": [target] 
            }
        self._identifiers=["symbol", "date"]
        self._metrics, self._standard_cols = self._load_configs()

        self._pro = MetricProcessor()
        self._pf = PriceFactorManager()

    def _load_configs(self):
        basepath = Path.cwd() / "config" / "fa_config"
        basepath.mkdir(parents=True, exist_ok=True)

        metrics = {}
        standard_cols = {}
        for domain in ["multiples", "price_factor"]:
            filepath = basepath / f"{domain}_add.json"
            with open(filepath, "r") as f:
                metrics[domain] = json.load(f)

            cols = ["symbol", "date"]
            for name, session_dict in metrics[domain].items():
                if "hidden" not in name:
                    cols.extend(session_dict.keys())
            standard_cols[domain] = cols
        return metrics, standard_cols

    def _fetch_dates(self, n=3):
        today = datetime.date.today()
        year = today.year

        past_dates = []
        y = year
        while len(past_dates) < n and y > 0:
            d = datetime.date(year=y, month=4, day=3)
            if d <= today:
                past_dates.append(d)
            y -= 1
        past_dates = sorted(past_dates)
        return past_dates + [today]
        
    def get_relative_valuation(self, fa, history, benchmark, n_dates=5, n_symbols=None): ##assumes fa and history are sorted (check_sortedness=False)
        symbols = fa["symbol"].unique(maintain_order=True).to_list()[:n_symbols]
        dates = self._fetch_dates(n_dates)
        df = pl.DataFrame({"symbol": symbols}).join(pl.DataFrame({"date": dates}), how="cross")

        fa = df.join_asof(fa.rename({"date": "d0"}), left_on="date", right_on="d0", by="symbol", strategy="backward", check_sortedness=False)
        fa = fa.with_columns(drift = (pl.col("date")-pl.col("d0")).dt.total_days())
        fa = fa.filter(pl.col("drift")<=365) #drop large drifts
        
        fa = fa.join_asof(history, on="date", by="symbol", strategy="backward", check_sortedness=False) 
        mt = self._pro.add_metrics(fa, self._metrics["multiples"]).select(self._standard_cols["multiples"]+["d0", "drift"]) #add q-q
       
        pf = self._pf.add_price_factors(df, history, benchmark, self._metrics["price_factor"])  ##pricexvolume
        rv = pf.join(mt, on=self._identifiers, how="inner")
        return rv

    """
    def to_cross_section(self, df, n_dates):        
        self.feat["num"] = sorted(list(set(df.columns) - set(self.feat["id"]+self.feat["cat"]+self.feat["target"])))
        all_dates = df["date"].unique().sort().to_list()
        valid_dates = df.drop_nulls(subset=self.feat["target"])["date"].unique().sort().to_list()[-n_dates:]
        dates = valid_dates + all_dates[all_dates.index(valid_dates[-1])+1:]
        print("All Dates: " + ", ".join(d.strftime("%Y-%m-%d") for d in dates))
        print("Train Dates: " + ", ".join(d.strftime("%Y-%m-%d") for d in valid_dates))
        df = df.filter(pl.col("date").is_in(dates))
        df = df.to_pandas()
        df["date"] = pd.to_datetime(df["date"]) 
        return df
    """        

class PriceFactorManager: 
    def __init__(self):
        self._identifiers = ["symbol", "date"]

    def _to_days(self, n, unit_type):
        mapping = {
            "d": 1,
            "w": 5,
            "mo": 21,
            "y": 252
        }
        if unit_type not in mapping:
            raise ValueError(f"Unsupported unit_type: {unit_type}. Use days, weeks, months, or years.")
        return int(abs(n) * mapping[unit_type])
        
    def _get_pct_change(self, df, history, params, col_name, col="adjClose", buffer=5):
        n0, n1, unit_type, sign = params
        df = df.select(self._identifiers)
        history = history.select(self._identifiers+[col])

        df = df.with_columns(
            t0 = (pl.col("date").dt.offset_by(f"{n0}{unit_type}")),
            t1 = (pl.col("date").dt.offset_by(f"{n1}{unit_type}"))
        )
        df = df.join_asof(history.rename({"date": "d0"}), left_on="t0", right_on="d0", by="symbol", strategy="backward", check_sortedness=False).rename({col: "p0"})
        df = df.join_asof(history.rename({"date": "d1"}), left_on="t1", right_on="d1", by="symbol", strategy="backward", check_sortedness=False).rename({col: "p1"})
        #t is theoretical date, d is actual date of the joined price
        df = df.with_columns([ 
            pl.when(
                pl.max_horizontal(
                    (pl.col("t0")-pl.col("d0")).dt.total_days(),
                    (pl.col("t1")-pl.col("d1")).dt.total_days()
                )<=buffer)
            .then(((pl.col("p1") / pl.col("p0")) - 1) * sign)
            .otherwise(None)
            .alias(col_name)
        ])
        return df.select(self._identifiers+[col_name])
    
    def _get_vol(self, df, history, params, col_name, min_vol_obs=120):
        n, unit_type = params
        vol_days = self._to_days(n, unit_type)
        df = df.select(self._identifiers)
        history = history.select(self._identifiers+["log_ret"])
        
        history = history.with_columns(
            pl.col("log_ret").rolling_std(window_size=vol_days, min_periods=min_vol_obs).over("symbol").alias(col_name)
        )   
        df = df.join_asof(history, on="date", by="symbol", strategy="backward", check_sortedness=False)
        return df.select(self._identifiers+[col_name])

    def _get_beta(self, df, history, benchmark, params, col_name, min_vol_obs=120, min_corr_obs=750):
        (n_vol, n_corr), unit_type, sign = params
        vol_days = self._to_days(n_vol, unit_type)
        corr_days = self._to_days(n_corr, unit_type)

        history = history.with_columns(
            asset_ret = pl.col("log_ret").rolling_sum(window_size=3).over("symbol"),
            asset_vol = pl.col("log_ret").rolling_std(window_size=vol_days, min_periods=min_vol_obs).over("symbol")
        ).select(["date", "symbol", "asset_ret", "asset_vol"])
        benchmark = benchmark.with_columns(
            bench_symbol = pl.col("symbol"),
            bench_ret = pl.col("log_ret").rolling_sum(window_size=3).over("symbol"),
            bench_vol = pl.col("log_ret").rolling_std(window_size=vol_days, min_periods=min_vol_obs).over("symbol")
        ).select(["date", "bench_symbol", "bench_ret", "bench_vol"])
        combined = history.join(benchmark, on="date", how="inner") #Cross: (N_assets x N_benchmarks)
        
        col_name_map = {c: f"{col_name}_{c}" for c in combined["bench_symbol"].unique()}
        central_beta = 1
        w = 0.67
        combined = combined.with_columns(
            corr = pl.rolling_corr(
                pl.col("asset_ret"),
                pl.col("bench_ret"),
                window_size=corr_days, min_periods=min_corr_obs
            ).over(["symbol", "bench_symbol"])
        )
        combined = combined.with_columns(raw_beta = pl.col("corr") * (pl.col("asset_vol") / pl.col("bench_vol")))
        combined = combined.with_columns(beta = (w * pl.col("raw_beta") + (1-w) * central_beta) * sign) #bayesian shrinkage
        
        combined = combined.unique(subset=["date", "symbol", "bench_symbol"]) ##why this is needed, prev smth wrong, smth very very wrong with everythings
        combined = combined.pivot( #LONG -> WIDE
            index = ["date", "symbol"],
            on = "bench_symbol",
            values = "beta",
            aggregate_function=None
        ).rename(col_name_map)
        
        df = df.join_asof(combined, on="date", by="symbol", strategy="backward", check_sortedness=False)
        return df.select(self._identifiers+list(col_name_map.values()))

    def add_price_factors(self, df, history, benchmark, price_config):
        history = history.with_columns(pl.col("adjClose").log().diff().over("symbol").alias("log_ret"))
        benchmark = benchmark.with_columns(pl.col("adjClose").log().diff().over("symbol").alias("log_ret"))
        benchmark = benchmark.select(["symbol", "date", "log_ret"])
                
        feature_cols = []
        for func, metrics in price_config.items():
            for col_name, params in metrics.items():
                if func == "price_change":
                    res = self._get_pct_change(df, history, params, col_name, col="adjClose")
                    feature_cols.append(res.drop(self._identifiers))
                    
                elif func == "vol":
                    res = self._get_vol(df, history, params, col_name)
                    feature_cols.append(res.drop(self._identifiers))

                elif func == "beta":
                    res = self._get_beta(df, history, benchmark, params, col_name)
                    feature_cols.append(res.drop(self._identifiers))
                    
        df = pl.concat([df]+feature_cols, how="horizontal")
        return df        
                    




        
