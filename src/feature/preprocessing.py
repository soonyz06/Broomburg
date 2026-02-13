import polars as pl
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split

from src.ml.feature.eda import display_nulls

def get_reciprocal(x, eps=1e-8):
    return 1.0 / (x + eps)

def numpy_transform(df, cols, func):
    df[cols] = func(df[cols])
    return df

def one_hot_encoding(df, cat):
    if isinstance(cat, str):
        cat = [cat]
    dummies = pd.get_dummies(df[cat], columns=cat, prefix=cat).astype(int)
    df_new = pd.concat([df, dummies], axis=1)
    return df_new, dummies.columns.tolist()

def label_encoding(df, cats):
    df = df.copy()
    new_cols = []
    for col in cats:
        new_col = col + "_le"
        df[new_col] = df[col].astype("category").cat.codes
        new_cols.append(new_col)
    return df, new_cols

class CategoricalEncoder:
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.stats_ = {}
    def fit(self, df, cats, y_cols):
        if isinstance(y_cols, str):
            y_cols = [y_cols]
        self.stats_ = {}
        for col in cats:
            self.stats_[col] = {}
            for y in y_cols:
                agg = df.groupby(col)[y].agg(self.strategy)
                self.stats_[col][y] = agg.to_dict()
        return self
    def transform(self, df, cats, y_cols=None, FIT=False):
        if FIT:
            if y_cols is None:
                raise ValueError("y_cols must be provided when FIT=True")
            self.fit(df, cats, y_cols)
        if not self.stats_:
            raise ValueError("Must fit before transform")
        if isinstance(y_cols, str):
            y_cols = [y_cols]
        df_new = df.copy()
        for col in cats:
            for y in y_cols:
                mapping = self.stats_[col][y]
                df_new[f"{col}_{y}_{self.strategy}_encoded"] = df_new[col].map(mapping)
        return df_new

class NeutraliseTransformer:
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.stats_ = None
    def fit(self, df, cols):
        self.stats_ = getattr(df[cols], self.strategy)()
        return self
    def transform(self, df, cols, FIT=True):
        if FIT:
            self.fit(df, cols)
        if self.stats_ is None:
            raise ValueError("Must fit before transform")
        df = df.copy()
        df[cols] = df[cols] - self.stats_
        return df

class WinsorTransformer:
    def __init__(self, alpha):
        self.upper_ = None
        self.lower_ = None
        self.alpha_ = alpha
        if self.alpha_ <= 0:
            raise ValueError("Invalid alpha")
    def fit(self, df, cols):
        self.upper_ = df[cols].quantile(1 - self.alpha_)
        self.lower_ = df[cols].quantile(self.alpha_)
        return self
    def transform(self, df, cols, FIT=True):
        if FIT: 
            self.fit(df, cols)
        if self.upper_ is None or self.lower_ is None:
            raise ValueError("Must fit before transform")
        df = df.copy()
        df[cols] = pd.DataFrame(df[cols].clip(self.lower_, self.upper_, axis=1), index=df.index, columns=cols)
        return df

class CustomTransformer:
    # StandardScaler, RobustScaler, QuantileTransformer
    # SimpleImputer, KNNImputer, IterativeImputer
    # PowerTransformer
    def __init__(self, transformer):
        self.transformer = transformer
    def fit(self, df, cols):
        if self.transformer is None:
            print("No transformer")
        else:
            self.transformer.fit(df[cols])
        return self
    def transform(self, df, cols, FIT=True):
        if self.transformer is None:
            print("No transformer")
            return df
        if FIT: 
            self.fit(df, cols)
        df[cols] = pd.DataFrame(self.transformer.transform(df[cols]), index=df.index, columns=cols)
        return df

def regime_transform(df, transform_fn, transformers):
    df_out = df.copy()
    for param in transform_fn:
        fn = param["fn"]
        group_sets = param["group_sets"]
        if "date" not in group_sets:
            group_sets = list(group_sets) + ["date"]

        groups = []
        for name, g in df_out.groupby(group_sets):
            if g.isna().all().any():
                print("Group", name, "has all-NaN column(s)")
            groups.append(fn(g, transformers, FIT=True))
        df_out = pd.concat(groups).reindex(df_out.index)

        if group_sets != ["date"]: #fallback
            groups = []
            for name, g in df_out.groupby(["date"]):
                if g.isna().all().any():
                    print("Group", name, "has all-NaN column(s) (date-only)")
                groups.append(fn(g, transformers, FIT=True))
            df_out = pd.concat(groups).reindex(df_out.index)
    return df_out

def pooled_transform(train_df, dfs, transform_fn, transformers):
    train_out = train_df.copy()
    for param in transform_fn:
        fn = param["fn"]
        group_sets = param["group_sets"]

        if not group_sets:
            train_out = fn(train_out, transformers, FIT=True)
        else:
            train_out = (
                train_out.groupby(group_sets, group_keys=False)[train_out.columns]
                .apply(lambda g: fn(g, transformers, FIT=True), include_groups=False)
                .reindex(train_out.index)
            )

    dfs_out = []
    for df in dfs:
        df_out = df.copy()
        for param in transform_fn:
            fn = param["fn"]
            group_sets = param["group_sets"]

            if not group_sets:
                df_out = fn(df_out, transformers, FIT=False)
            else:
                df_out = (
                    df_out.groupby(group_sets, group_keys=False)[df_out.columns]
                    .apply(lambda g: fn(g, transformers, FIT=False), include_groups=False)
                    .reindex(df_out.index)
                )
        dfs_out.append(df_out)
    return train_out, dfs_out

def features_construction(datasets, transformer, cols, name="transformed"):
    for i, df in enumerate(datasets):
        if i == 0:
            transformer.fit(df[cols])
        if hasattr(transformer, "get_feature_names_out"):
            new_cols = transformer.get_feature_names_out(cols)
        else:
            new_cols = [f"{col}_{name}" for col in cols]
        transformed_df = pd.DataFrame(transformer.transform(df[cols]), index=df.index, columns=new_cols)
        datasets[i] = pd.concat([df.drop(columns=cols), transformed_df], axis=1)
    return datasets
    
def get_summary(df, symbols, dates):
    print(f"Symbols: {len(symbols)}")
    print(f"Dates: {len(dates)}")
    print(f"Shape: {df.shape}")
    display_nulls(df, df.columns, 10)
    return df

def get_transformed_data(rv, feat, dates, transform_fn, transformers, POOL=True):
    feat["num"] = sorted(list(set(rv.columns) - set(feat["id"]+feat["cat"]+feat["dum"]+feat["target"])))
    if not POOL: rv = regime_transform(rv, transform_fn, transformers)
    train_df, test_df = train_test_split(rv[rv["date"]!=dates[-1]], test_size=0.2, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, shuffle=True)
    live_df = rv[rv["date"]==dates[-1]]
    if POOL: train_df, [val_df, test_df, live_df] = pooled_transform(train_df, [val_df, test_df, live_df], transform_fn, transformers)
    return [train_df, val_df, test_df, live_df]








    
