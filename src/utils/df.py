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

def max_metric(df, des, src):
    if des not in df.columns:
        df = df.with_columns(pl.col(src).alias(des))
    else:
        df = df.with_columns(
            pl.when(pl.col(des).is_null())
              .then(pl.col(src))
              .otherwise(pl.max_horizontal([pl.col(des), pl.col(src)]))
              .alias(des)
        )
    return df

def transpose_df(df, header):
    if header in df.columns:
        try:
            df = df.with_columns(pl.col(header).cast(pl.String))
            df = df.transpose(include_header=True, header_name="", column_names=header) 
        except Exception as e:
            print(f"Could not transpose by {header}: {e}")
    return df
