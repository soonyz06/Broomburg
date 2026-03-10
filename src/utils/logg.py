import time
from pathlib import Path
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

def log_info(msg, start=None):
    if start is None:
        print(f"[INFO]{msg}...")
        return time.perf_counter()
    else:
        elapsed = time.perf_counter() - start
        print(f"[INFO]{msg} done in {elapsed:.2f}s\n\n")
        return None
    
def log_csv(df, filename=None):
    if filename is None: return
    basepath = Path.cwd() / "data" / "output" 
    basepath.mkdir(parents=True, exist_ok=True)
    if Path(filename).suffix != ".csv": filename = filename + ".csv"

    if isinstance(df, pd.DataFrame):
        df.to_csv(basepath / filename)
    elif isinstance(df, pl.DataFrame):
        df.write_csv(basepath / filename)
    else:
        print("[WARNING]Invalid data")
        return
    print(df.head(3))

def log_plot(fig, SHOW=True): #flask, plotly?
    if SHOW:
        plt.show()
    else:
        plt.close()

def log_df(df, name, verbose=2):
    if df is None or verbose==0:
        return
    if verbose >=2:
        print(f"\n[INFO]{name}: {len(list(df['symbol'].unique()))}")
    else:
        print(f"\n[INFO]{name}")
    print(df.head(3))
    print(df.shape)
    if verbose>=3:
        print(df.columns)
   

