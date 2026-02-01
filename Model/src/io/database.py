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
        self._size_map = {'Mega Cap': 0, 'Large Cap': 1, 'Mid Cap': 2, 'Small Cap': 3, 'Micro Cap': 4, 'Nano Cap': 5, None: 6}

        self._writer = pm
        self._reader = pm
        
    def fetch_database(self, asset_type, REFRESH=False, COLLECT=False): 
        if asset_type not in self._config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        config = self._config[asset_type]
        
        filedir = self._basepath / f"{asset_type}" / "table"
        if not REFRESH:
            lf = self._reader.read_lazy(filedir, latest_by_identifiers=self._identifiers)
            if lf is not None:
                lf = lf.select(config["columns"])
                return lf.collect() if COLLECT else lf
        
        df = pl.from_pandas(config["fetcher"]().reset_index())
        df_final = df.select(config["columns"]).with_columns(pl.lit("fd").alias("source"))
        self._writer.save_parquet(filedir, df_final)
        return df_final if COLLECT else df_final.lazy()

    def filter_database(self, asset_type, filters={}, limit=None, REFRESH=False, COLLECT=True):
        lf = self.fetch_database(asset_type, REFRESH=REFRESH, COLLECT=False)
        lf = self._reader.filter_lazy(lf, filters=filters, COLLECT=False)

        exclude_folders = sorted(list(set(list(self._basepath.parent.glob("*"))) - {self._basepath}))
        dfs = [self._reader.read_lazy(f, latest_by_identifiers=self._identifiers) for f in exclude_folders]
        exlclude_lazy = pl.concat([d.select(self._identifiers+["asset_type", "error", "source", "timestamp"]) for d in dfs if d is not None]) #union of all exclude_dfs
        exclude_lazy = self._reader.filter_lazy(exclude_lazy, filters={"asset_type": asset_type, "error": None}, inclusive=True) #union of all excludes (by error)
        if exclude_lazy is None:
            lf = self._n_lazy(lf, asset_type, limit, COLLECT=COLLECT)
            return lf, None
        lf = lf.join(exclude_lazy.select(self._identifiers), on="symbol", how="anti")
        lf = self._n_lazy(lf, asset_type, limit, COLLECT=COLLECT)
        exclude_lazy = exclude_lazy.limit(limit)
        return lf, exclude_lazy.collect() if COLLECT else exclude_lazy
    
    def _n_lazy(self, lf, asset_type, limit, COLLECT=True):
        if asset_type == "equities":
            lf = lf.with_columns(pl.col("market_cap").replace(self._size_map).alias("size")).sort(["size", "symbol"], nulls_last=True).drop("size")
        lf = lf.limit(limit)
        return lf.collect() if COLLECT else lf
