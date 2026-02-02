import datetime
from pathlib import Path
import polars as pl
from yahooquery import Ticker
import json


def get_highlights(df=None, domains=["income_statement", "cash_flow", "balance_sheet"]):
    cols = ["date", "symbol"]
    rename = []
    for domain in domains:
        file_path = Path.cwd() / "data" / "fa_config" / f"{domain}_add.json"
        with open(file_path, "r") as f:
            metrics = json.load(f)
                   
        for name, session_dict in metrics.items():
            if "hidden" not in name:
                cols.extend(session_dict.keys())
    if df is None:
        return cols
    df = df.with_columns([pl.lit(0).alias(c) for c in cols if c not in df.columns])
    df = df.select(list(dict.fromkeys(cols)))
    return df

def save_model_data(n_symbols, n_dates, identifiers):
    base_path = file_path = Path.cwd() / "data" / "staging"    
    t0 = time.perf_counter()
    symbols = fetch_symbols(n_symbols)
    dates = fetch_dates(n_dates)

    fa, background = preload_data(symbols) 
    df = pl.DataFrame({"symbol": symbols}).join(pl.DataFrame({"date": dates}), how="cross")
    fa = df.join_asof(fa, on="date", by="symbol", strategy="backward", check_sortedness=False)
    #fill null bs ?
    print(f"[INFO]Preloaded in {time.perf_counter() - t0:.2f} seconds")

    t1 = time.perf_counter()
    rv = get_relative_valuation_data(fa, background)
    print(f"[INFO]Retrieved RV in {time.perf_counter() - t1:.2f} seconds")
    
    t2 = time.perf_counter()
    fa = get_highlights(fa)
    fa_cols = [c for c in fa.columns if c not in rv.columns or c in identifiers]
    rv = rv.join(fa.select(fa_cols), on=identifiers, how="left")
    print(f"[INFO]Retrieved FA in {time.perf_counter() - t2:.2f} seconds")
    rv.write_parquet(base_path / f"PF(a).parquet")
    return rv

def load_model_data():
    base_path = file_path = Path.cwd() / "data" / "staging"
    rv = pl.read_parquet(base_path / f"PF(a).parquet")
    return rv

def get_model_data(feat, n_symbols, n_dates, domains=["income_statement", "cash_flow", "balance_sheet", "relative_valuation", "price_factor"], keys=["symbol"], LOAD=True):
    rv = load_model_data() if LOAD else save_model_data(n_symbols, n_dates+2, feat["id"])

    if domains !=None:
        cols = get_highlights(df=None, domains=domains)
        rv = rv.select(cols+feat["cat"])
    rv =  rv.drop_nulls(subset=keys).sort(keys, descending=True)

    all_dates = rv["date"].unique().sort().to_list()
    valid_dates = rv.drop_nulls(subset=feat["target"])["date"].unique().sort().to_list()[-n_dates:]
    dates = valid_dates + all_dates[all_dates.index(valid_dates[-1])+1:]
    rv = rv.filter(pl.col("date").is_in(dates))
    
    symbols = rv["symbol"].unique(maintain_order=True).to_list()[:n_symbols]
    dates = sorted(rv["date"].unique().sort(descending=True).to_list())
    rv = rv.filter((rv["symbol"].is_in(symbols)) & (rv["date"].is_in(dates)))
    print("All Dates: " + ", ".join(d.strftime("%Y-%m-%d") for d in dates))
    print("Train Dates: " + ", ".join(d.strftime("%Y-%m-%d") for d in valid_dates))
    print(f"Symbols: {len(symbols)}")

    feat["num"] = sorted(list(set(rv.columns) - set(feat["id"]+feat["cat"]+feat["target"])))
    rv = rv.select(feat["id"]+feat["num"]+feat["cat"]+feat["target"])
    print(rv[[c for c in rv.columns if c in ["date", "MC", "HML_5", "RevGrowth"]+feat["target"]]].filter(pl.col("date")==dates[0]).head(3))
    print(rv[[c for c in rv.columns if c in ["date", "MC", "HML_5", "RevGrowth"]+feat["target"]]].filter(pl.col("date")==dates[0]).tail(3))
    return rv, symbols, dates, feat
    
def get_initial_data(n_symbols, n_dates, domains, keys=["symbol"], LOAD=False):
    feat = {}
    feat["id"] = ["symbol", "date"]
    feat["num"] = []
    feat["cat"] = ["country", "sectorKey", "industryKey"]
    feat["dum"] = []
    feat["target"] = ["returns"] #nextmc
    
    rv, symbols, dates, feat = get_model_data(feat, n_symbols, n_dates, domains, keys, LOAD)
    
    feat["cat"] = ["symbolKey", "countryKey"] + feat["cat"][1:]
    rv = rv.with_columns(pl.col("symbol").alias("symbolKey"))
    rv = rv.rename({"country": "countryKey"})
    rv = rv.to_pandas()
    rv["date"] = pd.to_datetime(rv["date"])
    return rv, feat, pd.to_datetime(dates), symbols

def append_parquet(filedir, df_new, source, identifiers, file_limit_mb=512): 
    if df_new is None or df_new.is_empty():
        return 
    assert isinstance(df_new, pl.DataFrame), "Should be polars dataframe"
    filedir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.datetime.now(datetime.timezone.utc) #versioning: ignores duplication across chunks
    df_new = df_new.with_columns(pl.lit(ts).cast(pl.Datetime("us", "UTC")).alias("timestamp"))
    if "source" not in df_new.columns:
        df_new = df_new.with_columns(pl.lit(source).alias("source"))

    existing_files = [f for f in os.listdir(filedir) if f.endswith(".parquet")] 
    if existing_files: 
        
        latest_file = existing_files[-1]
        latest_num = int(latest_file.split("_")[0])
        latest_path = filedir / latest_file

        size_mb = os.path.getsize(latest_path) / (1024*1024)
        if size_mb < file_limit_mb:  # Append
            df_old = pl.read_parquet(latest_path)
        else:  # Overflow
            df_old = None
            latest_num += 1
    else:  # New
        df_old = None
        latest_num = 1

    ordered_cols = identifiers + [c for c in df_new.columns if c not in identifiers]
    df_new = df_new.select(ordered_cols)
    if df_old is None:
        df_combined = df_new
    else:
        df_old = df_old.select(ordered_cols)
        df_combined = pl.concat([df_old, df_new])
    df_combined.write_parquet(filedir / f"{latest_num}_chunk.parquet")
    return df_combined

