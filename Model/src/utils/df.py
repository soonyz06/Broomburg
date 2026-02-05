import polars as pl

def replace_metric(df, des, src):
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
