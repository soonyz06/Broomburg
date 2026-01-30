from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import os
import datetime
import re

import gcsfs

from src.utils.logg import log_info

def synch_gcs():
    t0 = log_info("Initialise GCS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path.cwd() / "config" / "gcs.json")
    fs = gcsfs.GCSFileSystem(project="your-gcp-project-id")
    gcspath = None #"your-bucket-name/path/to/output.parquet"
    log_info("Initialise GCS", t0)
    return gcspath # #with fs.open(gcs_path, "wb") as f:

class Writer:
    def __init__(self, max_size=512*1024*1024, batch_size=100_000):
        self.MAX_SIZE = max_size
        self.batch_size = batch_size
        
        self.idx = 0
        self.current_size = 0
        
    def get_rg_size(self, arrow_batch, speed_mode=1):
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
        
    def append_parquet(self, filedir, df, source, identifiers, FORCENEW=False, backup_dir=None):
        if df is None or df.is_empty():
            return self
        assert isinstance(df, pl.DataFrame), "Should be a polars dataframe"
        filedir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc) #versioning: ignores duplication across chunks

        #Backup
        if backup_dir is None:
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
            self.idx = int(last_file.split("_")[1].split(".")[0])
            current_path = filedir / last_file
            self.current_size = os.path.getsize(current_path)
        else:
            self.idx = 1
            current_path = filedir / f"chunk_{self.idx:04d}.parquet"
            self.current_size = 0
            writer = pq.ParquetWriter(current_path, arrow_schema) #first
            
        for start in range(0, df.height, self.batch_size):
            arrow_batch = df[start:start+self.batch_size].to_arrow()
            rg_size = self.get_rg_size(arrow_batch, speed_mode=1)

            #Increment Writer
            if self.current_size + rg_size > self.MAX_SIZE or FORCENEW:
                if FORCENEW: #for versioning during compact job
                    print("[INFO]Starting from chunk_{self.idx}") 
                    FORCENEW = False 
                if writer is not None:
                    writer.close()
                self.idx += 1
                current_path = filedir / f"chunk_{self.idx:04d}.parquet"
                self.current_size = 0
                writer = pq.ParquetWriter(current_path, arrow_schema) #overflow

            #Write
            if writer is None: #IsAppend=True
                existing = pq.read_table(current_path)
                existing = existing.drop(partitions)
                combined = pa.concat_tables([existing, arrow_batch]) #append
                pq.write_table(combined, current_path)
                print(f"[INFO]Appended {rg_size//1024}KB to chunk_{self.idx}")
            else:
                writer.write_table(arrow_batch)
                print(f"[INFO]Written {rg_size//1024}KB to chunk_{self.idx}")
            self.current_size += rg_size
            n += rg_size
        if writer is not None:
            writer.close()
        #print(f"[INFO]Total Data Written: {n//1024}KB")
        return self

    def cleanup_backups(self, backup_dir, now, max_age_hours=24):
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
    
