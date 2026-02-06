import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logg import log_info
from src.feature.preprocessing import get_summary ,\
#     get_initial_data, get_transformed_data, 
 #    one_hot_encoding, label_encoding, get_reciprocal,\
  #   numpy_transform, NeutraliseTransformer, WinsorTransformer, CustomTransformer, CategoricalEncoder, \
   #  features_construction

"""
from src.feature.eda import missing_heatmap, Custom_EDA
from src.feature.decomposition import Custom_Decomposition
from src.model.config import make_rf, make_xgb
from src.feature.embedding import learned_embedding, visualise_embeddings

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, \
     SplineTransformer, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA, SparsePCA, FastICA, DictionaryLearning
"""


#-----Params-----
n_symbols = 20
n_dates = 5
top_k = 2
basepath = Path.cwd() / "data" / "output" / "models"
basepath.mkdir(parents=True, exist_ok=True)

group_sets = []
"""
transformers = {
    "imputer": CustomTransformer(SimpleImputer(strategy="median")), #other imputrers
    "transformer": CustomTransformer(PowerTransformer(method="yeo-johnson", standardize=True)),
    "winsor": WinsorTransformer(alpha=0.05),
    "neutraliser": NeutraliseTransformer(strategy="median"),
    "scaler": CustomTransformer(RobustScaler()), #QuantileTransformer(output_distribution="uniform")
    "encoder": CategoricalEncoder(strategy="count") #target/freq/agg encoding
}
spline = SplineTransformer(degree=3, n_knots=5, include_bias=False) 
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
reciprocal = FunctionTransformer(get_reciprocal)
"""

#-----Fetch-----
t0 = log_info("Fetch")
rv, feat, dates, symbols = get_initial_data(n_symbols, n_dates, domains=["price_factor"], keys=["symbol"], LOAD=True)
rv = get_summary(rv, symbols, dates)
print(rv.head()) #validate mc is right, etc (TRC)


stop
"""
transform_fn = [
    {"fn": lambda df, transformers, FIT=True:(
        transformers["imputer"].transform(df, feat["num"], FIT)), "group_sets": ["sectorKey"]},
    {"fn": lambda df, transformers, FIT=True:(
        transformers["winsor"].transform(df, feat["num"]+feat["target"], FIT) #winsor vs transform target
        .pipe(lambda d: transformers["transformer"].transform(d, feat["num"], FIT))), "group_sets": []}, 
    {"fn": lambda df, transformers, FIT=True:(
        transformers["encoder"].transform(df, feat["cat"], feat["target"], FIT)
        .pipe(lambda d: transformers["scaler"].transform(d, sorted(list(set(d.columns) - set(feat["id"]+feat["cat"]+feat["dum"]+feat["target"]))), FIT))), "group_sets": []} 
]
"""s


datasets = get_transformed_data(rv, feat, dates, transform_fn, transformers, POOL=False)  
rv = get_summary(pd.concat(datasets, axis=0), symbols, dates)

stop
datasets = features_construction(datasets, reciprocal, feat["num"], "1/x") ###create new features->NAN, interaction terms(*/top), missing flags
datasets = get_transformed_data(pd.concat(datasets, axis=0), feat, dates, transform_fn, transformers, POOL=True)

##update feat after each transformed (encoding) and new features :), encode instead of demean




stop
datasets, lookup_dict = learned_embedding(datasets, feat, params=None)
clusters = visualise_embeddings(lookup_dict, model=PCA() ,n_dim=3, POOL=False)

feat["num"] = sorted(list(set(datasets[0].columns) - set(feat["id"]+feat["cat"]+feat["dum"]+feat["target"])))
features, target = feat["num"]+feat["dum"], feat["target"]
print(f"Features: {features}")
print(f"Shape: {train_df.shape}")
log_info("Fetch", t0)


#-----EDA----- 
name = "EDA1" 
t0 = log_info(name) 

#missing_heatmap(rv, SHOW=True)
df = pd.concat([train_df, val_df, test_df], axis=0)
#eda = Custom_EDA(df, feat, top_k)
#eda.md_qq_plot()
#eda.hausman_test()
#eda.categorical_dist(1)
#eda.numerical_dist(2)
#eda.feature_corr()
#eda.target_dist().target_corr()
#eda.regression(SHOW=False)

eda2 = Custom_Decomposition(df, features=feat["num"], model=PCA()) 
#eda2.parallel_analysis()
#eda2.loadings()
#eda2.biplot(top_k=top_k, colour="returns")
#scores_df = eda2.transform()
clusters = eda2.network_plot(n_dim=2, names=["AAPL", "NVDA", "TSM"]) 


stop
#-----Training-----
t0 = log_info("Training")
model = make_rf()
[[X_train, y_train], [X_val, y_val], [X_test, y_test], [X_live, y_live]] = model.get_training_data(datasets, features, target)
model.fit(None, [X_train, y_train, X_val, y_val], basepath, LOAD=False)
model.results_summary()#.error_diag().SHAP_diag(top_k, SHOW=False).explanation_diag(["AAPL"])
top_features = model.get_top_features(top_k)
print(f"Top {top_k} Features: {top_features}")
log_info("Training", t0)



#-----Help----- (use best existing solutions)
#dict, class, asserts
#state + attributes + method (solution?)
#asserts are defenses for things that should never happen (not to validate) (assumptions)


#{Inference + Prediction}
#R_{i,t}-R_{f,t} = a_{i} + B_{i,1}F_{1,t} + ... + B_{i,k}F_{k,t} + e_{i,t} (factor loading)
#regression of asset returns on factor returns (time-series regression)
#R_{i,t}-R_{f,t} = λ_{0,t} + λ_{1,t}B_{i,1} + λ_{k,t}B_{k,1} + n_{i,t} (risk premia)
#regression of asset returns on factors characteristics (cross-sectional regression)


##eda, data cleaning, preprocessing, feature construction, eda, feature selection, modelling
#reg, bagging, nn, imputer, FT Transformer, GNN
#documentation
#encde+mi+splits+shap (construction)
#rfecv+dropout+pca+regularisation (selection)
#isolation forest for outliers 
#factor mimicking portfolios (measure factor returns), vol-targetting, attribution


#nlp for cli
#raw text + other -> staging (RAG) -> LLCM/MCP -> inference: summaries/query/search_engine (nline vs offline)
#RAG -> chunk, embed, index (query~solution contrast loss + cosine similarity)


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

