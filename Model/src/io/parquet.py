from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import os
import datetime
import re
import shutil
#import gcsfs

from src.utils.logg import log_info


def synch_gcs():
    t0 = log_info("Initialise GCS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path.cwd() / "config" / "gcs.json")
    fs = gcsfs.GCSFileSystem(project="your-gcp-project-id")
    gcspath = "your-bucket-name/path/to/output.parquet"
    #with fs.open(gcs_path, "wb") as f:
    log_info("Initialise GCS", t0)
    return gcspath 

class ParquetManager:
    def __init__(self, max_file_size=512*1024*1024, max_rg_size=128*1024*1024, batch_size=100_000):
        self._MAX_FILE_SIZE = max_file_size
        self._MAX_RG_SIZE = max_rg_size
        self._batch_size = batch_size
        assert self._MAX_FILE_SIZE >= self._MAX_RG_SIZE, "Max File Size should be >= Max Row Group"
        
        self._backup_dir = Path.cwd() / "data" / "backup"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def _clear_folder(self, ts, filedir):
        filedir.mkdir(parents=True, exist_ok=True)
        domain = filedir.name
        run_backup_dir = self._backup_dir / f"{domain}_backup_{ts.strftime('%Y%m%dT%H%M%S%fZ')}"
        run_backup_dir.mkdir(parents=True, exist_ok=True)
        for f in filedir.rglob("chunk_*.parquet"):
            if f.exists():
                dest = run_backup_dir / f.name
                print(f"Moving {f} -> {dest}")
                shutil.move(f, dest)
        return self

    def _cleanup_backups(self, max_age_hours=24):
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(hours=max_age_hours)

        for item in self._backup_dir.glob("*_backup_*"):
            try:
                ts_str = item.name.split("_backup_")[-1]
                ts = datetime.datetime.strptime(ts_str, "%Y%m%dT%H%M%S%fZ")
                ts = ts.replace(tzinfo=datetime.timezone.utc)
                if ts < cutoff:
                    if item.is_file():
                        item.unlink(missing_ok=True)
                        print(f"[INFO]Deleted old backup file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"[INFO]Deleted old backup folder: {item.name}")
            except Exception as e:
                print(f"[WARNING]Failed to delete backup: {item.name} due to {e}")
        return self
    
    def _get_rg_size(self, arrow_batch, accuracy=1):
        if accuracy<=0: 
            rg_size = arrow_batch.nbytes #arrow batch heuristic
        elif accuracy == 1: 
            sink = pa.BufferOutputStream() #buffer-based
            pq.write_table(arrow_batch, sink)
            buf = sink.getvalue()
            rg_size = buf.size
        else:
            tmp_path = "tmp.parquet" #file-based
            pq.write_table(arrow_batch, tmp_path)
            rg_size = os.path.getsize(tmp_path) 
            os.remove(tmp_path)
        return rg_size

    def _save_subtable(self, filedir, df, _BACKUP=False): 
        if df is None or df.is_empty():
            return self
        assert isinstance(df, pl.DataFrame), "Should be a polars dataframe"
        assert "source" in df.columns, "DataFrame should have a 'source' column"
        filedir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc) 

        if _BACKUP:
            backup_file = self._backup_dir / f"{domain}_backup_{ts.strftime('%Y%m%dT%H%M%S%fZ')}.parquet"
            #print(f"Writing DataFrame -> {backup_file}")
            df.write_parquet(backup_file)

        df = df.with_columns(pl.lit(ts).cast(pl.Datetime("us", "UTC")).alias("timestamp")) 
        arrow_schema = df.to_arrow().schema
        files = sorted([f for f in os.listdir(filedir) if f.endswith(".parquet")]) #can be directly sorted due to :04d
        idx = int(files[-1].split("_")[1].split(".")[0]) if files else 0
        
        idx += 1
        current_path = filedir / f"chunk_{idx:04d}.parquet"
        current_size = 0
        writer = pq.ParquetWriter(current_path, arrow_schema) #ParquetWriter for each sub-directory

        start = 0
        batch_size = self._batch_size #assumes rg_size of each batch is approx the same
        total_size = 0
        while start < df.height:
            batch_df = df[start:start+batch_size]    
            arrow_batch = batch_df.to_arrow()
            rg_size = self._get_rg_size(arrow_batch, accuracy=1)

            while rg_size > self._MAX_RG_SIZE and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                arrow_batch = df[start:start+batch_size].to_arrow()
                rg_size = self._get_rg_size(arrow_batch, accuracy=1)
            assert self._MAX_RG_SIZE >= rg_size, "MAX Row Group Size should be > Row Group Size"

            if current_size + rg_size > self._MAX_FILE_SIZE: #overflow
                if writer is not None:
                    writer.close()
                idx += 1
                current_path = filedir / f"chunk_{idx:04d}.parquet"
                current_size = 0
                writer = pq.ParquetWriter(current_path, arrow_schema) #ignores duplication across chunks, requires scheduling compact-jobs
    
            writer.write_table(arrow_batch) 
            #print(f"[INFO]Written {rg_size//1024}KB to chunk_{idx}")
            current_size += rg_size
            total_size += rg_size
            start += batch_size
        if writer is not None:
            writer.close()
        print(f"[INFO]Written a total of {total_size//1024}KB to {filedir}")
        return self

    def _scan_subtable(self, filedir, latest_by_identifiers): 
        filedir.mkdir(parents=True, exist_ok=True) 
        files = [f for f in filedir.glob("chunk_*.parquet") if f.stat().st_size >= 12] #maybe this works? idk :)
        if not files: return None

        lf = pl.scan_parquet(files) #assumes same schema
        
        if latest_by_identifiers is not None:
            latest = lf.group_by(latest_by_identifiers).agg(pl.col("timestamp").max().alias("timestamp")) #lazy filtering: deduplicate at read
            lf = latest.join(lf, on=latest_by_identifiers+["timestamp"], how="inner") #faster than global sort + unique, but uses all latest rows per group instead of only one

        if lf is not None:
            df = lf.collect()
        #print(f"[INFO]Scanned {filedir}")
        return lf

    def read_lazy(self, filedir, latest_by_identifiers): #handles partition pruning from filedir
        filedir.mkdir(parents=True, exist_ok=True) 
        files = list(filedir.rglob(f"chunk_*.parquet"))
        filedirs = sorted({f.parent for f in files}) #set literal, sorted for reproducibility
        #print(f"[INFO]Reading {filedirs} from {filedir}")

        lfs = []
        for subdir in filedirs:
            lf = self._scan_subtable(subdir, latest_by_identifiers)
            if lf is None:
                continue

            matches = []
            for part in Path(subdir).parts:
                if "=" in part:
                    col, key = part.split("=", 1)
                    matches.append((col, key))        
            #print(f"Matches: {matches} from {str(subdir)}")
            lfs.append(lf.with_columns([pl.lit(key).alias(col) for col, key in matches])) #virtual cols
        partition_cols = [col for col, _ in matches] if lfs else [] #not useful if partition pruning is applied as alr known 
        lf = pl.concat(lfs) if lfs else None #partitioning assumes same schema
        return lf, partition_cols

    def save_parquet(self, filedir, df, partition_cols, _BACKUP=False):     
        if partition_cols is None or not partition_cols:
            self._save_subtable(filedir, df, _BACKUP=_BACKUP)
            return self
            
        assert set(partition_cols).issubset(df.columns), "Partition cols should be present in the DataFrame"
        subtables = df.partition_by(partition_cols, as_dict=True) #split into sub-tables
        for keys, subdf in subtables.items():
            subdir = filedir.joinpath(*[f"{col}={key}" for col, key in zip(partition_cols, keys)])
            subdf = subdf.drop(partition_cols) #dropped at write and added at read to save memory (virtual col)
            self._save_subtable(subdir, subdf, _BACKUP=_BACKUP)
        return self

    def compact_job(self, filedir): #merging and deduplication 
        lf, partition_cols = self.read_lazy(filedir, latest_by_identifiers=None) #None -> versioning
        ts = datetime.datetime.now(datetime.timezone.utc) 
        self._cleanup_backups(max_age_hours=24)
        if lf is None or lf.limit(1).collect().is_empty(): return self
        df = lf.collect()
        self._clear_folder(ts, filedir) 
        self.save_parquet(filedir, df, partition_cols) #holds df in memory, peforms R/W operations in batches to reduce io overhead
        return self

    def filter_lazy(self, lf, filters=None, inclusive=True, COLLECT=False): #in isolation, aggeragate -> join (semi, anti)
        if lf is None:
            return None
        
        filters = filters or {}
        schema_names = lf.collect_schema().names()
        for key, val in filters.items():
            if key not in schema_names:   
                print(f"[WARNING]{key} is an invalid filter key")
                continue
            if val is None:
                continue
            val = val if isinstance(val, list) else [val]
            lf = lf.filter(pl.col(key).is_in(val)) if inclusive else lf.filter(pl.col(key).is_in(val).not_())            
        return lf.collect() if COLLECT else lf

    def sample_df(self, df, sampling_rate=None, seed=42):
        if df is None:
            return None 
        assert isinstance(database, pl.DataFrame), "Should be polars dataframe"
        
        if sampling_rate is None or sampling_rate < 0:
            return df
        elif sampling_rate <= 1:
            return df.sample(fraction=sampling_rate, seed=seed)
        else:
            return df.sample(n=min(sampling_rate, df.height), seed=seed)





