from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np
import json
import io
import tarfile
import zstandard as zstd
from tqdm import tqdm

from edgar import set_identity, Company, get_current_filings
from edgar.xbrl import XBRL

from src.io.fetch import validate_date
from src.io.manager import ManagerManager

    
class EdgarParser:
    def __init__(self, sleep_s):
        #https://edgartools.readthedocs.io/en/latest/api/filing/?h=filing#filingcik-company-form-filing_date-accession_no
        set_identity("soon.yz061025@gmail.com")
        self._sleep_s = sleep_s
        with open(Path.cwd() / "config" / "edgartools" / "display_names.json") as f:
            self._standard_concepts = list(json.load(f).keys())
        self._content_limit = 3

    def fetch_current_filings(self, forms, page_size=100, latency_s=3):
        current = get_current_filings(form=forms, page_size=page_size) #math.ceil(100/page_size) HTTP requests per call
        #current = get_filings(form=forms).latest(100) #1 HTTP request per filing
        for filing in current:
            name = {"name": filing.company, "form": {filing.form}, "date": filing.filing_date, "accession": filing.accession_number, "url": filing.filing_url}
            for key, val in name.items():
                print(f"{key.title()}: {val}")
            print("")
            time.sleep(latency_s)
        return self

    def process_statements(self, filings, symbol, forms): #form_key
        if not isinstance(forms, list): forms = [forms]
        if not set(forms).issubset({"10-K", "20-F", "10-Q", "6-K"}): return None
        
        output = None
        for filing in filings:
            time.sleep(self._sleep_s)
            _key = {"symbol": symbol, "date": filing.filing_date}

            try:
                symbol = _key["symbol"]
                xbrl = XBRL.from_filing(filing)
                #xbrl = filing.xbrl() #immediate loading
                stmts = xbrl.statements
                stmts_map = {
                    "IS": stmts.income_statement(), 
                    "BS": stmts.balance_sheet(),
                    "CF": stmts.cashflow_statement(),
                    #"CI": stmts.comprehensive_income(),
                    #"ES": stmts.statement_of_equity() 
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

    def process_obj(self, filings, symbol, forms):
        if not isinstance(forms, list): forms = [forms]
        if not set(forms).issubset({"10-K", "20-F"}): return None
        
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

    def process_content(self, filings, symbol, func, filedir): ## parse xrbl and text using llm 
        mode = func[:-1]
        assert isinstance(mode, str), "Mode should be a string"
        if len(list(filings))>1:
            filings = filings.latest(self._content_limit)
        
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for filing in filings:
                time.sleep(self._sleep_s)
                _key =  {"symbol": symbol, "date": filing.filing_date}
                content = None
                
                try:
                    if mode == "text":
                        content = filing.text()
                        name = f"date={filing.filing_date}.txt"
                    elif mode == "markdown":
                        content = filing.markdown()
                        name = f"date={filing.filing_date}.md"
                    elif mode == "html":
                        content = filing.html()
                        name = f"date={filing.filing_date}.html"
                    else:
                        raise ValueError("Invalid mode")
                except Exception as e:
                    print(f"[WARNING]Failed to fetch {mode} ({_key}) due to {e}")
                    content = None
        
                if content is not None:
                    data = content.encode("utf-8")
                    fileobj = io.BytesIO(data)
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    tar.addfile(info, fileobj=fileobj)
        
        tar_buffer.seek(0)
        cctx = zstd.ZstdCompressor(level=3) #better alternatives?
        compressed = cctx.compress(tar_buffer.read())
        ts = datetime.datetime.now(datetime.timezone.utc) 
        tardir = filedir / func
        tardir.mkdir(parents=True, exist_ok=True)
        tarpath = tardir / f"{symbol}_{ts.strftime('%Y%m%dT%H%M%S%fZ')}.tar.zst"
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
        self.forms =  {
            "annual": ["10-K", "20-F"],
            "quarterly": ["10-Q", "6-K"],
            "current": ["8-K"],
            "insiders": ["3", "4", "5"],
            "ownership": ["13D", "13F", "13G"],
            "registration": ["S-1"],
            "proxy": ["DEF 14A"],

            "annual_US": ["10-K"],
            "annual_IN": ["20-F"],
            "quarterly_US": ["10-Q"],
            "quarterly_IN": ["6-K"],
            "insider_initial": ["3"],
            "insider_change": ["4"],
            "insider_annual": ["5"],
            "ownership_institutional": ["13F"],
            "ownership_active": ["13D"],
            "ownership_passive": ["13G"]
            }
        self._cacheable_funcs = ["statements", "sections"]

        self._manager = mm
        self._writer = self._manager._writer
        self._reader = self._manager._reader
        self._sources = sources
        
        self._edgar = EdgarParser(0.5) #:)

    def _load_batch(self, asset_type, form_key, funcs, batch_df, exclude_lazy, filedir, partition_map, refresh_threshold_days, REFRESH): 
        assert batch_df.columns[0] == "symbol", "First column of each batch_df should be 'symbol'"
        lf_map = {func:
                  (self._reader.read_lazy(
                      filedir.joinpath(func, *[f"{col}={val}" for col, val in partition_map.items()]), 
                      latest_by_identifiers=self._identifiers)[0])
                      if func in self._cacheable_funcs else None
                  for func in funcs}
        forms = self.forms[form_key]
        
        batch_success_map = {func: [] for func in funcs}
        batch_failed = []
        for row in batch_df.iter_rows(named=False):
            time.sleep(self._sleep_s)
            startDate = self._history_start
            symbol=row[0]
            
            company = Company(symbol)  
            filings = company.get_filings( #1 HTTP request per symbol (json containing every filing, regardless of date)
                form=forms,
                filing_date=f"{startDate}:", 
                amendments=False
            ) #maps cuz tryna reuse this, idk what i'm doing :)

            for func in funcs:
                startDate = self._history_start
                
                #SKIP -> Already Cached
                if not REFRESH:
                    days_elapsed, startDate = self._manager.get_days_elapsed(lf_map[func], filters={"symbol": [symbol]}, startDate=startDate) #only checks where trying to load (subdir), as only filters by symbol)
                    if days_elapsed is not None and (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
                        continue
                
                #FETCH -> Symbol
                if startDate is not None and startDate != self._history_start:
                    filings = filings.filter(filing_date=f"{startDate}:")
                
                for source, api_key in self._sources.items():
                    time.sleep(self._sleep_s)
                    df_new = None
                    
                    #Skip -> Already Failed
                    if exclude_lazy is not None:
                        IsInExclude = not self._reader.filter_lazy(
                            exclude_lazy,
                            filters={"symbol": [symbol], "asset_type": [asset_type], "form_key": [form_key], "func": [func], "source": [source]}, inclusive=True
                            ).limit(1).collect().is_empty()
                        if IsInExclude:
                            continue 

                    #FETCH -> Source
                    try:
                        print(f"[INFO]Fetching {symbol}: {startDate}")
                        if source == "edgar": 
                            if func == "statements":
                                df_new = self._edgar.process_statements(filings, symbol, forms)
                            elif func ==  "sections":
                                df_new = self._edgar.process_obj(filings, symbol, forms)
                            elif func in ["texts", "markdowns", "htmls"]:
                                df_new = self._edgar.process_content(filings, symbol, func, filedir) #on demand (not cached)
                            else:
                                print(f"[WARNING]{source} doesn't have func '{func}'")
                        else:
                            raise ValueError(f"Unknown source '{source}'")
                        
                        if df_new is None:
                            raise ValueError(f"'{source}': {func} is empty")
                        elif isinstance(df_new, pl.DataFrame):
                            if df_new.is_empty():
                                raise ValueError(f"'{source}': {func} is empty")
                            else:
                                df_new = df_new.with_columns(pl.lit(source).alias("source"))
                                batch_success_map[func].append(df_new)
                        print(f"[INFO]Successfully fetched {symbol}")
                        break 
                    except Exception as e:
                        print(f"[WARNING]Error fetching {symbol}: {e}")
                        df_new = {"symbol": symbol, "asset_type": asset_type, "form_key": form_key, "func": func, "source": source, "error": str(e).split("\n")[0][:100]}
                        batch_failed.append(df_new)

        for func, batch_success in batch_success_map.items():
            if batch_success:
                if func not in self._cacheable_funcs: 
                    continue
                funcdir = filedir.joinpath(func, *[f"{col}={val}" for col, val in partition_map.items()])
                batch_success = pl.concat(batch_success)
                self._writer.save_parquet(funcdir, batch_success, partition_cols=None) #alr pruned 
        return batch_failed

    def load_sec(self, database, form_key="annual", funcs=["statements", "sections"], refresh_threshold_days=None, REFRESH=False, asset_type = "equities"):
        if database is None or database.is_empty():
            return None, None, None
        partition_cols, database, exclude_lazy, exclude_dir = self._manager.initialise_manager(asset_type, database, self._basepath)
        database = database.select(["symbol"]+[c for c in database.columns if c != "symbol"])
        symbols = list(database["symbol"].unique())
        filedir = self._basepath / asset_type / form_key 
        params = {"asset_type": asset_type, "form_key": form_key, "funcs": funcs, "filedir": filedir, \
                  "exclude_lazy": exclude_lazy, "refresh_threshold_days": refresh_threshold_days, "REFRESH": REFRESH}
        
        batch_idx = 0
        total_symbols = len(database)
        total_calls = total_symbols*len(funcs)
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
                batch_failed = self._load_batch(**params, batch_df=batch_df, partition_map=partition_map) 
                failed_calls.extend(batch_failed)
        del batch_df, batch_failed

        success_rates = {}
        success_rates = {"calls": [(total_calls - len(failed_calls)), total_calls]}    
        if failed_calls:
            failed_df = pl.DataFrame(failed_calls)
            failed_symbols = failed_df["symbol"].n_unique()
            success_rates["symbols"] = [(total_symbols - failed_symbols), total_symbols]
            self._writer.save_parquet(exclude_dir, failed_df, partition_cols=None)
            for func in funcs:
                failed_func = failed_df.filter(pl.col("func") == func)["symbol"].n_unique()
                success_rates[func] = [(total_symbols - failed_func),  total_symbols] 
        else:
            failed_df = None
            success_rates["symbols"] = [(total_symbols), total_symbols]
            
        for key, val in success_rates.items():
            success_rate = (val[0] / val[1]) * 100 if val[1] > 0 else np.nan
            print(f"[INFO]Successfully fetched: {success_rate:.2f}% of {key} ({val[1]})") 

        lf_map = {func:
                  (self._reader.read_lazy(
                      filedir.joinpath(func, *[f"{col}={val}" for col, val in partition_map.items()]), 
                      latest_by_identifiers=self._identifiers)[0])
                      if func in self._cacheable_funcs else None
                  for func in funcs}
        output_map = {func: self._reader.filter_lazy(lf, filters={"symbol": symbols}, COLLECT=True) for func, lf in lf_map.items()} 
        return output_map, partition_cols, failed_df

    def compact_job(self, asset_type, form_key, funcs): 
        filedir = self._basepath / asset_type / form_key
        for func in funcs:
            self._writer.compact_job(filedir / func)
        return self





