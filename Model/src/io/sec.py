from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np
import json
import io
import tarfile
import zstandard as zstd

from edgar import set_identity, Company, get_current_filings
#from edgar.xbrl import XBRL, XBRLS

from src.io.fetch import validate_date
from src.io.manager import ManagerManager


class EdgarParser:
    def __init__(self, sleep_s):
        ##https://edgartools.readthedocs.io/en/latest/api/filing/?h=filing#filingcik-company-form-filing_date-accession_no
        set_identity("soon.yz061025@gmail.com")
        self._sleep_s = sleep_s
        with open(Path.cwd() / "config" / "edgartools" / "display_names.json") as f:
            self._standard_concepts = list(json.load(f).keys())

    def fetch_current_filings(self, form, page_size=100, latency_s=3):
        current = get_current_filings(form=form, page_size=page_size) #math.ceil(100/page_size) HTTP requests per call
        #current = get_filings(form=form).latest(100) #1 HTTP request per filing
        for filing in current:
            name = {"name": filing.company, "form": {filing.form}, "date": filing.filing_date, "accession": filing.accession_number, "url": filing.filing_url}
            for key, val in name.items():
                print(f"{key.title()}: {val}")
            print("")
            time.sleep(latency_s)
        return self

    def process_statements(self, filings, symbol, form):
        if form not in ["10-K", "20-F", "10-Q", "6-K"]: return None
        
        output = None
        for filing in filings:
            time.sleep(self._sleep_s)
            _key = {"symbol": symbol, "date": filing.filing_date}

            try:
                symbol = _key["symbol"]
                #xbrl = XBRL.from_filing(filing)
                xbrl = filing.xbrl() #immediate loading
                stmts = xbrl.statements
                stmts_map = {
                    "IS": stmts.income_statement(), 
                    "BS": stmts.balance_sheet(),
                    "CF": stmts.cashflow_statement(),
                    "CI": stmts.comprehensive_income(),
                    "ES": stmts.statement_of_equity() 
                }
                
                stmt_df = None
                for key, val in stmts_map.items(): 
                    df = val.to_dataframe().reset_index(drop=True) 
                    df = df.replace("", None)
                    df = pl.from_pandas(df) ##for deeper breakdown save here :)
                    if df.is_empty():
                        continue

                    dates = [d for d in df.columns if validate_date(d, verbose=0)]
                    cols = ["standard_concept"]+dates
                    df = (
                        df
                        .drop_nulls(subset=cols)
                        .with_columns([(pl.col(col)*pl.col("weight")).alias(col) for col in dates])
                        .select(cols) 
                        .group_by("standard_concept")
                        .agg([pl.sum(col).alias(col) for col in dates])
                        )
                    cols = df["standard_concept"].to_list()
                    df = df.drop("standard_concept")
                    assert set(df.columns).issubset(set(dates)), "Exists a non-date column that has not yet been dropped"
                    if df.is_empty():
                        continue

                    df_t = df.transpose()
                    df_t.columns = cols
                    df_t = df_t.with_columns(pl.lit(symbol).alias("symbol"))
                    df_t = df_t.with_columns(pl.Series("date", df.columns))
                    df_t = df_t.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")) #cast "date" from str -> date 

                    if stmt_df is None or stmt_df.is_empty():
                        stmt_df = df_t
                    else:
                        cols = set(stmt_df.columns) | set(cols) 
                        stmt_df = stmt_df.join(df_t, how="left", on=list(_key.keys())).select(cols) 
                    
                cols = list(_key.keys()) + self._standard_concepts
                if not set(stmt_df.columns).issubset(set(cols)):
                    print(f"[WARNING]Columns of statements ({_key}) is not a subset of total standard concepts")
                stmt_df = stmt_df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in cols if c not in stmt_df.columns]).select(cols)
            except Exception as e:
                print(f"[WARNING]Failed to fetch statements ({_key}) due to {e}")
                stmt_df = None
            
            if stmt_df is not None and not stmt_df.is_empty():
                output = pl.concat([output, stmt_df], how="vertical") if output is not None else stmt_df
        return output

    def process_obj(self, filings, symbol, form): 
        if form not in ["10-K", "20-F"]: return None
        
        output = []
        for filing in filings:
            time.sleep(self._sleep_s)
            _key = {"symbol": symbol, "date": filing.filing_date, "company": filing.company}
            
            sections = ["business", "risk_factors", "management_discussion"]
            try:
                report = filing.obj() #uses lazy loading
                row = _key | {"accession": filing.accession_number}
                for section_name in sections:
                    if hasattr(report, section_name):
                        row[section_name] = getattr(report, section_name)
                    else:
                        row[section_name] = None
                section_row = row
            except Exception as e:
                print(f"[WARNING]Failed to fetch section ({_key}) due to {e}")
                section_row = None
        
            if section_row is not None:
                output.append(section_row)
        return pl.DataFrame(output)

    def process_content(self, filings, symbol, partitioneddir, func): ##fix idk, cuz returned val is not df? help me and parse xrbl and text using llm -> RAG
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for filing in filings:
                time.sleep(self._sleep_s)
                _key =  {"symbol": symbol, "date": filing.filing_date}
                mode = func[:-1]
                
                assert isinstance(mode, str), "Mode should be a string"
                try:
                    mode = mode.lower()
                    if mode == "text":
                        raw = filing.text()
                    elif mode == "markdown":
                        raw = filing.markdown()
                    elif mode == "html":
                        raw = filing.html()
                    else:
                        raise ValueError("Invalid mode")
                    content = raw            
                except Exception as e:
                    print(f"[WARNING]Failed to fetch {mode} ({_key}) due to {e}")
                    content = None
        
                if content is not None:
                    for c in content:
                        data = c.encode("utf-8")
                        name = f"date={filing.filing_date}.txt"
                        fileobj = io.BytesIO(data)
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        tar.addfile(info, fileobj=fileobj)

        tar_buffer.seek(0)
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(tar_buffer.read())
        tarpath = partitioneddir / f"symbol={symbol}.tar.zst"
        with open(tarpath, "wb") as f:
            f.write(compressed)
        return tarpath
    
class SECManager:
    def __init__(self, mm: ManagerManager, sources, history_start):
        self._basepath = Path.cwd() / "data" / "raw" / "sec" 
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._history_start = history_start #dry run
        self._identifiers = ["symbol", "date"]
        self._sleep_s = 0.2
        self._batch_size = 100 

        self._manager = mm
        self._writer = self._manager._writer
        self._reader = self._manager._reader
        self._sources = sources
        
        self._edgar = EdgarParser(0.2) #:)

    def _load_batch(self, asset_type, form, funcs, batch_df, exclude_lazy, filedir, partition_map, refresh_threshold_days, REFRESH): 
        assert batch_df.columns[0] == "symbol", "First column of each batch_df should be 'symbol'"
        lf_map = {func:
                  self._reader.read_lazy(
                      filedir.joinpath(func, *[f"{col}={val}" for col, val in partition_map.items()]), #func outside partition_cols
                      latest_by_identifiers=self._identifiers)[0]
                  for func in funcs}
        
        cacheable_funcs = ["statements", "sections"] #cache vs on demand 

        batch_success_map = {func: [] for func in funcs}
        batch_failed = []
        for row in batch_df.iter_rows(named=False):
            time.sleep(self._sleep_s)
            startDate = self._history_start
            symbol=row[0]
            
            company = Company(symbol)  
            filings = company.get_filings( #1 HTTP request per symbol (json containing every filing, regardless of date)
                form=form,
                filing_date=f"{startDate}:", 
                amendments=False
            ) #maps cuz tryna reuse this, idk what i'm doing :)

            for func in funcs:
                print(f"Function: {func}")
                #SKIP -> Already Cached
                if not REFRESH:
                    days_elapsed, startDate = self._manager.get_days_elapsed(lf_map[func], filters={"symbol": [symbol]}) #only checks where trying to load (subdir), as only filters by symbol)
                    if days_elapsed is not None and (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
                        continue
                
                #FETCH -> Symbol
                if startDate is not None and startDate != self._history_start:
                    filings = filings.filter(filing_date=f"{startDate}:")
                print(filings)
                
                for source, api_key in self._sources.items():
                    time.sleep(self._sleep_s)
                    df_new = None
                    
                    #Skip -> Already Failed
                    if exclude_lazy is not None:
                        IsInExclude = not self._reader.filter_lazy(
                            exclude_lazy,
                            filters={"symbol": [symbol], "asset_type": [asset_type], "form": [form], "func": [func], "source": [source]}, inclusive=True
                            ).limit(1).collect().is_empty()
                        if IsInExclude:
                            continue 

                    #FETCH -> Source
                    try:
                        print(f"[INFO]Fetching {symbol}: {startDate}")
                        if source == "edgar": 
                            if func == "statements":
                                df_new = self._edgar.process_statements(filings, symbol, form)
                            elif func ==  "sections":
                                df_new = self._edgar.process_obj(filings, symbol, form)
                            elif func in ["texts", "markdowns", "htmls"]:
                                df_new = self._edgar.process_content(filings, symbol, partitioneddir, func)
                            else:
                                print(f"[WARNING]{source} doesn't have func '{func}'")
                        else:
                            raise ValueError(f"Unknown source '{source}'")
                        if df_new is None or df_new.is_empty():
                            raise ValueError(f"'{source}' is empty")
                        df_new = df_new.with_columns(pl.lit(source).alias("source"))
                        batch_success_map[func].append(df_new)
                        print(f"[INFO]Successfully fetched {symbol}")
                        break 
                    except Exception as e:
                        print(f"[WARNING]Error fetching {symbol}: {e}")
                        df_new = {"symbol": symbol, "asset_type": asset_type, "form": form, "func": func, "source": source, "error": str(e).split("\n")[0][:100]}
                        batch_failed.append(df_new)

        for func, batch_success in batch_success_map.items():
            if batch_success:
                if func not in cacheable_funcs: 
                    continue
                funcdir = filedir.joinpath(func, *[f"{col}={val}" for col, val in partition_map.items()])
                batch_success = pl.concat(batch_success)
                self._writer.save_parquet(funcdir, batch_success, partition_cols=None) #alr pruned 
        return batch_failed

    def load_sec(self, database, form="10-K", funcs=["statements", "sections"], refresh_threshold_days=None, REFRESH=False, asset_type = "equities"):
        partition_cols, database, exclude_lazy, exclude_dir = self._manager.initialise_manager(asset_type, database, self._basepath)
        database = database.select(["symbol"]+[c for c in database.columns if c != "symbol"])
        filedir = self._basepath / asset_type / form
        params = {"asset_type": asset_type, "form": form, "funcs": funcs, "filedir": filedir, \
                  "exclude_lazy": exclude_lazy, "refresh_threshold_days": refresh_threshold_days, "REFRESH": REFRESH}


        batch_idx = 0 
        total_calls = len(database)*len(funcs)
        failed_calls = []        
        print(f"\n[INFO]Fetching SEC Data ({total_calls})")
        assert set(partition_cols).issubset(database.columns), "Partition cols should be present in the DataFrame"
        subtables = database.partition_by(partition_cols, as_dict=True)
        for keys, subdf in subtables.items():
            partition_map = {col: val for col, val in zip(partition_cols, keys)}
            for offset in range(0, subdf.height, self._batch_size):
                print(f"[INFO]Batch {batch_idx}")
                batch_idx +=1
                batch_df = subdf.slice(offset, self._batch_size)
                ##this func is abstractable for mm using _load_batch(**params: dict)
                batch_failed = self._load_batch(**params, batch_df=batch_df, partition_map=partition_map) ###func should be outside partition
                failed_calls.extend(batch_failed)
        del batch_df, batch_failed

        success_rate = ((total_calls - len(failed_calls)) / total_calls) * 100 if total_calls>0 else np.nan
        print(f"[INFO]Successfully fetched: {success_rate:.2f}% of symbols ({total_calls})") ##could breakdown per func, change price to func too?

        symbols = list(database["symbol"].unique())
        lf_map = {func:  self._reader.read_lazy(filedir / func, latest_by_identifiers=self._identifiers)[0] for func in funcs}
        output_map = {func: self._reader.filter_lazy(lf, filters={"symbol": symbols}, COLLECT=True) for func, lf in lf_map.items()}
        
        failed_df = pl.DataFrame(failed_calls) if failed_calls else None
        self._writer.save_parquet(exclude_dir, failed_df, partition_cols=None) 
        return output_map, partition_cols, failed_df

##df check None and .is_empty()
#read: input latest_by_identifiers and output partition_cols
#save: input df requires partition_cols in df.columns and source, it adds partition_cols to filedir
#sleep: source, row





