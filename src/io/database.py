from pathlib import Path
import polars as pl

from config.config import ASSET_CONFIGS
from src.io.parquet import ParquetManager


class DatabaseManager:
    def __init__(self):
        self._config = ASSET_CONFIGS
        self._basepath = Path.cwd() / "data" / "raw" / "database"
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._identifiers = ["symbol"] #const
        
        self._writer = ParquetManager()
        self._reader = ParquetManager()
        
    def fetch_database(self, asset_type, filters=None, REFRESH=False):
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
                lf = lf.with_columns([pl.col(c).str.replace_all(" ", "_").str.to_lowercase() #.str.replace_all(r"[^a-zA-Z0-9._-]", "")
                                      for c in [s for s in selected_columns if s != "symbol"]])
                lf = self._reader.filter_lazy(lf, filters=filters, COLLECT=False)
                lf = lf.unique(subset=self._identifiers, maintain_order=True)
                return lf

        #FETCH -> Asset Type
        df = pl.from_pandas(config["fetcher"]().reset_index())
        df_final = df.select(selected_columns)
        df_final = df_final.with_columns(pl.lit("fd").alias("source"))
        self._writer.save_parquet(filedir, df_final, partition_cols=None)
        lf = df_final.lazy()
        lf = lf.with_columns([pl.col(c).str.replace_all(" ", "_").str.to_lowercase() #.str.replace_all(r"[^a-zA-Z0-9._-]", "").
                                      for c in [s for s in selected_columns if s != "symbol"]])
        lf = self._reader.filter_lazy(lf, filters=filters, COLLECT=False)
        lf = lf.unique(subset=self._identifiers, maintain_order=True)
        return lf

    def equity_filter(self, lf, asset_type):
        if asset_type != "equities":
            return lf
        lf = lf.filter(~pl.col("symbol").str.contains(r"[^A-Za-z]"))
        return lf
