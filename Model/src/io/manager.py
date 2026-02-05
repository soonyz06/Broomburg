import datetime
import polars as pl

from config.config import ASSET_CONFIGS
from src.io.parquet import ParquetManager

class ManagerManager:
    def __init__(self, pm: ParquetManager):
        self._config = ASSET_CONFIGS
        self._today = datetime.date.today()
        self._partition_degree = 2 #overhead (n1 x n2 x n3)
        self._writer = pm
        self._reader = pm
        
    def get_days_elapsed(self, lf, filters, startDate):
        df_old = self._reader.filter_lazy(lf, filters=filters, COLLECT=True) 
        exist_bool = not (df_old is None or df_old.is_empty())
        if exist_bool and "date" in df_old.columns:
            maxDate = df_old["date"].max()
            startDate = maxDate.strftime("%Y-%m-%d") 
            days_elapsed = (self._today - maxDate).days
            return days_elapsed, startDate
        return None, startDate

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

