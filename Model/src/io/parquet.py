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
    gcspath = None #"your-bucket-name/path/to/output.parquet"
    log_info("Initialise GCS", t0)
    return gcspath # #with fs.open(gcs_path, "wb") as f:

class ParquetManager:
    def __init__(self, max_size_mb=512, batch_size=100_000):
        self._MAX_SIZE = max_size_mb*1024*1024
        self._batch_size = batch_size
        
        self._idx = 0
        self._current_size = 0

    def _clear_folder(self, filedir):
        filedir.mkdir(parents=True, exist_ok=True)
        backup_dir = Path.cwd() / "data" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for f in filedir.rglob("chunk_*.parquet"):
            if f.exists():
                dest = backup_dir / f.name
                print(f"Moving {f} -> {dest}")  
                shutil.move(f, dest) 
        return backup_dir

    def _cleanup_backups(self, backup_dir, now, max_age_hours=24):
        cutoff = now - datetime.timedelta(hours=max_age_hours)
        for f in backup_dir.glob("*.parquet"):
            try:
                ts = datetime.datetime.strptime(f.stem.replace("backup_", ""), "%Y%m%dT%H%M%S%fZ").replace(tzinfo=datetime.timezone.utc) 
                if ts < cutoff:
                    f.unlink(missing_ok=True)
                    print(f"[INFO]Deleted old backup: {f.name}")
            except Exception as e:
                print(f"[WARNING]Failed to delete backup: {f.name}")
        return self
    
    def _get_rg_size(self, arrow_batch, speed_mode=1):
        if speed_mode<=0: #file-based
            tmp_path = "tmp.parquet"
            pq.write_table(arrow_batch, tmp_path)
            rg_size = os.path.getsize(tmp_path) 
            os.remove(tmp_path)
        elif speed_mode == 1: #buffer-based
            sink = pa.BufferOutputStream()
            pq.write_table(arrow_batch, sink)
            buf = sink.getvalue()
            rg_size = buf.size
        else:
            rg_size = arrow_batch.nbytes #arrow batch heuristic
        return rg_size
        
    def append_parquet(self, filedir, df, identifiers, source, APPEND=False, BACKUP=False, CLEAR=False): 
        if df is None or df.is_empty():
            return self
        assert isinstance(df, pl.DataFrame), "Should be a polars dataframe"
        filedir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc) #versioning: ignores duplication across chunks

        if BACKUP:
            backup_dir = Path.cwd() / "data" / "backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup_backups(backup_dir, now=ts, max_age_hours=24)
            df.write_parquet(backup_dir / f"backup_{ts.strftime('%Y%m%dT%H%M%S%fZ')}.parquet")
        
        #Processing
        df = df.with_columns(pl.lit(ts).cast(pl.Datetime("us", "UTC")).alias("timestamp"))
        if "source" not in df.columns:
            df = df.with_columns(pl.lit(source).alias("source"))
        df = df.select(identifiers + [c for c in df.columns if c not in identifiers])
        segments = filedir.parts
        
        partitions = []
        for seg in reversed(segments): #read_table adds partitions as new cols, which fks up alignment with batch_arrow :)
            m = re.match(r'(\w+)=(\w+)', seg)
            if m:
                partitions.append(m.groups()[0])
            else:
                break
        
        #Initialise Writer
        n = 0
        arrow_schema = df.to_arrow().schema
        files = sorted([f for f in os.listdir(filedir) if f.endswith(".parquet")])
        if files:
            last_file = files[-1]
            self._idx = int(last_file.split("_")[1].split(".")[0])
            if APPEND: 
                current_path = filedir / last_file
                self._current_size = os.path.getsize(current_path)
                writer = None
            else:
                self._idx += 1
                current_path = filedir / f"chunk_{self._idx:04d}.parquet"
                self._current_size = 0
                writer = pq.ParquetWriter(current_path, arrow_schema)
        else: #first
            self._idx = 1
            current_path = filedir / f"chunk_{self._idx:04d}.parquet"
            self._current_size = 0
            writer = pq.ParquetWriter(current_path, arrow_schema)
            
        for start in range(0, df.height, self._batch_size):
            arrow_batch = df[start:start+self._batch_size].to_arrow()
            rg_size = self._get_rg_size(arrow_batch, speed_mode=1)

            #Increment Writer
            if self._current_size + rg_size > self._MAX_SIZE:
                if writer is not None:
                    writer.close()
                self._idx += 1
                current_path = filedir / f"chunk_{self._idx:04d}.parquet"
                self._current_size = 0
                writer = pq.ParquetWriter(current_path, arrow_schema) #overflow

            if CLEAR:
                self._clear_folder(filedir) 
                CLEAR = False
                
            #Write
            if writer is None: #IsAppend=True
                existing = pq.read_table(current_path)
                existing = existing.drop(partitions)
                combined = pa.concat_tables([existing, arrow_batch]) #append
                pq.write_table(combined, current_path)
                print(f"[INFO]Appended {rg_size//1024}KB to chunk_{self._idx}")
            else:
                writer.write_table(arrow_batch)
                print(f"[INFO]Written {rg_size//1024}KB to chunk_{self._idx}")
            self._current_size += rg_size
            n += rg_size
        if writer is not None:
            writer.close()
        #print(f"[INFO]Total Data Written: {n//1024}KB")
        return self

    def compact_job(self, filedir, identifiers): #merging and deduplication 
        lf = self.read_lazy(filedir, identifiers)
        self.append_parquet(filedir, df, source=None, identifiers=identifiers, CLEAR=True) #(file-level versioning)
        return

    def read_lazy(self, filedir, identifiers, filename=None, cols=None): 
        filedir.mkdir(parents=True, exist_ok=True)
        if filename is None:        
            files = list(filedir.rglob(f"chunk_*.parquet"))
        else:
            files = list(filedir.rglob(f"*/{filename}/chunk_*.parquet"))
        if not files:
            return None

        if cols is None:
            lf = pl.scan_parquet(files)
        else:
            lf= pl.concat([pl.scan_parquet(f).select(cols) for f in files])
        if identifiers is None:
            return lf
            
        latest = lf.group_by(identifiers).agg(pl.col("timestamp").max().alias("timestamp")) #lazy filtering: deduplicate at read
        lf = latest.join(lf, on=identifiers+["timestamp"], how="inner") #faster than global sort + unique, but uses all latest rows per group instead of only one
        return lf

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
