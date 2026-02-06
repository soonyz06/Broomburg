import datetime
import polars as pl

from config.config import ASSET_CONFIGS
from src.io.parquet import ParquetManager

class ManagerManager: #delete one day :)
    def __init__(self):
        self._config = ASSET_CONFIGS
        self._today = datetime.date.today()
        self._partition_degree = 2 #overhead (n1 x n2 x n3)
        self._writer = ParquetManager()
        self._reader = ParquetManager()

    def initialise_manager(self, asset_type, database, basepath):
        assert isinstance(database, pl.DataFrame), "Should be polars dataframe"
        if asset_type not in self._config:
            raise ValueError(f"Asset type '{asset_type}' not supported.")
        if database is None or database.is_empty():
            print("[WARNING]Invalid database")
            return None, None, None, None
        assert set(self._config[asset_type]["columns"]).issubset(database.columns), "DataFrame should be a database from DM"

        partition_cols = self._config[asset_type]["partitions"][:self._partition_degree] #frequently filtered + low cardinality
        database = database.with_columns([pl.col(c).cast(pl.Utf8).fill_null("Unknown") for c in partition_cols])
        exclude_dir = basepath / "exclude" 
        exclude_lazy, _ = self._reader.read_lazy(exclude_dir, latest_by_identifiers=["symbol"])
        return partition_cols, database, exclude_lazy, exclude_dir

