import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info, log_df
from src.io.api import load_apikeys
from src.io.database import DatabaseManager
from src.io.fundamental import FundamentalManager
from src.io.price import PriceManager
from src.feature.cross_section import CrossSectionManager
from src.utils.df import transpose_df

#GOAL: Collect Information
#Simplest, Best at each step, ml oritented, lean+simple, 


limit = 5
api_keys = load_apikeys()
history_start = "1980-01-01"
dm = DatabaseManager()


t0 = log_info("Benchmark")
asset_type = "etfs"
benchmarks = ["ACWI", "SPY", "GLD"]
database = dm.fetch_database(asset_type=asset_type, filters=None) #fix asset_type so derived instead
database = dm._reader.filter_lazy(database, filters={"symbol": benchmarks})
database = database.collect()
log_df(database, "DM loaded")
print("-"*120)

frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None), "yahooquery": None}
pm = PriceManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
benchmark, failed_df = pm.load_history(asset_type, database, frequency, refresh_threshold_days=5, REFRESH=False) #threshold
#log_df(benchmark, "BM loaded", verbose=3)
log_df(failed_df, "BM excludes")
print("-"*120)
log_info("Benchmark", t0)


t0 = log_info("Asset")
asset_type = "equities"
filters = {"currency": ['usd'], "exchange": ['nms', 'nyq', 'ngm', 'ncm', 'ase', 'pcx', 'pnk'], "market_cap": ["mega_cap"]}
database = dm.fetch_database(asset_type=asset_type, filters=filters) #fix asset_type so derived instead
database = dm.equity_filter(database, asset_type)
database = database.limit(limit).collect()
log_df(database, "DM loaded")
print("-"*120)


frequency = "daily"
sources = {"tiingo": api_keys.get("TIINGO_API_KEY", None), "yahooquery": None}
pm = PriceManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
history, failed_df = pm.load_history(asset_type, database, frequency, refresh_threshold_days=5, REFRESH=False) #threshold
#log_df(history, "PM loaded", verbose=3)
log_df(failed_df, "PM excludes")
print("-"*120)

frequency = "annual"
sources = {"yahooquery": None}
fm = FundamentalManager(sources=sources, history_start=history_start)#.compact_job(asset_type, frequency=frequency)
fa, failed_df = fm.load_fundamentals(asset_type, database, frequency=frequency, refresh_threshold_days=None, REFRESH=False)
#log_df(fa, "FM loaded")
log_df(failed_df, "FM excludes")
print("-"*120)
log_info("Asset", t0)


t0 = log_info("Cross Section")
cm = CrossSectionManager()
rv = cm.get_relative_valuation(fa, history, benchmark, n_dates=5)
rv.write_csv("rv.csv")
log_df(rv, "RV loaded")
print("-"*120)
log_info("Cross Section", t0)


t0 = log_info("Modelling")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, \
     SplineTransformer, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA, SparsePCA, FastICA, DictionaryLearning

from src.feature.preprocessing import CustomTransformer, WinsorTransformer, NeutraliseTransformer, CategoricalEncoder, \
     get_transformed_data
from src.feature.embedding import learned_embedding, visualise_embeddings

#-----Params-----
n_symbols = 20
n_dates = 5
top_k = 2
basepath = Path.cwd() / "data" / "output" / "models"
basepath.mkdir(parents=True, exist_ok=True)

transformers = {
    "imputer": CustomTransformer(SimpleImputer(strategy="median")), #other imputrers
    "transformer": CustomTransformer(PowerTransformer(method="yeo-johnson", standardize=True)),
    "winsor": WinsorTransformer(alpha=0.05),
    "neutraliser": NeutraliseTransformer(strategy="median"),
    "scaler": CustomTransformer(RobustScaler()), #QuantileTransformer(output_distribution="uniform")
    "encoder": CategoricalEncoder(strategy="count") #target/freq/agg encoding
}
transform_fn = [
    {"fn": lambda df, transformers, FIT=True:(
        transformers["imputer"].transform(df, feat["num"], FIT)), "group_sets": ["sector"]},
    {"fn": lambda df, transformers, FIT=True:(
        transformers["winsor"].transform(df, feat["num"]+feat["target"], FIT) #winsor vs transform target
        .pipe(lambda d: transformers["transformer"].transform(d, feat["num"], FIT))), "group_sets": []}, 
    {"fn": lambda df, transformers, FIT=True:(
        transformers["encoder"].transform(df, feat["cat"], feat["target"], FIT)
        .pipe(lambda d: transformers["scaler"].transform(d, sorted(list(set(d.columns) - set(feat["id"]+feat["cat"]+feat["target"]))), FIT))), "group_sets": []} 
]

rv = pl.read_csv("rv.csv")
rv = cm.get_model_data(rv, database, n_dates=5)
rv = rv.fillna(0)
dates = rv["date"].unique().tolist()
feat = cm.get_feat()
print(rv.head())
print("-"*120)

datasets = get_transformed_data(rv, feat, dates, transform_fn, transformers, POOL=False)  
datasets, lookup_dict = learned_embedding(datasets, feat, params=None)
print(lookup_dict)
clusters = visualise_embeddings(lookup_dict, model=PCA() ,n_dim=2, POOL=False)
feat["num"] = sorted(list(set(datasets[0].columns) - set(feat["id"]+feat["cat"]+feat["target"])))
features, target = feat["num"], feat["target"]
print(f"Features: {features}")
print(f"Shape: {train_df.shape}")


        
log_info("Modelling", t0)






