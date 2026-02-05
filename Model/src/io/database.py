from pathlib import Path
import polars as pl

from config.config import ASSET_CONFIGS
from src.io.parquet import ParquetManager


class DatabaseManager:
    def __init__(self, pm: ParquetManager):
        self._config = ASSET_CONFIGS
        self._basepath = Path.cwd() / "data" / "raw" / "database"
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._identifiers = ["symbol"] #const
        
        self._writer = pm
        self._reader = pm
        
    def fetch_database(self, asset_type, filters=None, limit=None, REFRESH=False, COLLECT=False):
        if asset_type not in self._config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        config = self._config[asset_type]
        selected_columns = config["columns"]
        filedir = self._basepath / f"{asset_type}" / "table"

        #SKIP -> Already Cached
        if not REFRESH:
            lf, partition_cols = self._reader.read_lazy(filedir, latest_by_identifiers=self._identifiers)
            if lf is not None:
                lf = lf.select(selected_columns)
                lf = lf.with_columns([pl.col(c).str.replace_all(" ", "_").str.replace_all(r"[^a-zA-Z0-9._-]", "").str.to_lowercase()
                                      for c in [s for s in selected_columns if s != "symbol"]])
                lf = self._reader.filter_lazy(lf, filters=filters, COLLECT=False)
                lf = lf.unique(subset=self._identifiers, maintain_order=True).limit(limit)
                return lf.collect() if COLLECT else lf

        #FETCH -> Asset Type
        df = pl.from_pandas(config["fetcher"]().reset_index())
        df_final = df.select(selected_columns)
        df_final = df_final.with_columns(pl.lit("fd").alias("source"))
        self._writer.save_parquet(filedir, df_final, partition_cols=None)
        lf = df_final.lazy()
        lf = lf.with_columns([pl.col(c).str.replace_all(" ", "_").str.replace_all(r"[^a-zA-Z0-9._-]", "").str.to_lowercase()
                                      for c in [s for s in selected_columns if s != "symbol"]])
        lf = self._reader.filter_lazy(lf, filters=filters, COLLECT=False)
        lf = lf.unique(subset=self._identifiers, maintain_order=True).limit(limit)
        return lf.collect() if COLLECT else lf

    def filter_excludes(self, lf, asset_type, filters=None, limit=None, COLLECT=False):
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        assert isinstance(lf, pl.LazyFrame), "lf should be a LazyFrame"
        if filters is None: filters = {"asset_type": asset_type}

        #UNION Excludes
        exclude_folders = sorted([f for f in self._basepath.parent.glob("*/exclude") if f.is_dir()])
        dfs = [self._reader.read_lazy(f , latest_by_identifiers=self._identifiers)[0] for f in exclude_folders]
        inner_dfs = [d.select(self._identifiers+["asset_type", "error", "source", "timestamp"]) for d in dfs if d is not None]
        exclude_lazy = pl.concat(inner_dfs) if inner_dfs else None
        exclude_lazy = self._reader.filter_lazy(exclude_lazy, filters=filters, inclusive=True) #filter 
        if exclude_lazy is None:
            if asset_type == "equities":
                _size_map = {'Mega Cap': 0, 'Large Cap': 1, 'Mid Cap': 2, 'Small Cap': 3, 'Micro Cap': 4, 'Nano Cap': 5, None: 6}
                lf = lf.with_columns(pl.col("market_cap").replace(_size_map).alias("size")).sort(["size", "symbol"], nulls_last=True).drop("size")
            lf = lf.limit(limit)
            return lf.collect() if COLLECT else lf, None

        #FILTER BY Excludes
        lf = lf.join(exclude_lazy.select(self._identifiers), on="symbol", how="anti")
        if asset_type == "equities":
            _size_map = {'Mega Cap': 0, 'Large Cap': 1, 'Mid Cap': 2, 'Small Cap': 3, 'Micro Cap': 4, 'Nano Cap': 5, None: 6}
            lf = lf.with_columns(pl.col("market_cap").replace(_size_map).alias("size")).sort(["size", "symbol"], nulls_last=True).drop("size")
        lf = lf.limit(limit)
        return lf.collect() if COLLECT else lf, exclude_lazy.collect() if COLLECT else exclude_lazy




    
