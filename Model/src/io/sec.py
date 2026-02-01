from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np
import gzip
import json
import zstandard as zstd
    
from edgar import set_identity, Company, get_current_filings
#from edgar.xbrl import XBRL, XBRLS

from config.config import ASSET_CONFIGS
from src.io.fetch import validate_date
from src.io.parquet import ParquetManager


class SECManager:
    def __init__(self, pm: ParquetManager, sources):
        self._config = ASSET_CONFIGS
        self._basepath = Path.cwd() / "data" / "raw" / "sec" 
        self._basepath.mkdir(parents=True, exist_ok=True)
        self._history_start = "2022-01-01" #dry run
        self._today = datetime.date.today()
        self._identifiers = ["symbol", "date"]
        self._partition_degree = 1
        self._sleep_s = 0.2
        self._cctx = zstd.ZstdCompressor(level=3)
        
        set_identity("soon.yz061025@gmail.com")
        with open(Path.cwd() / "config" / "edgartools" / "display_names.json") as f:
            self._standard_concepts = list(json.load(f).keys())
                
        self._writer = pm
        self._reader = pm
        self._sources = sources 

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
    
    def _filing_stmts(self, identifiers, filing):
        time.sleep(self._sleep_s)
        try:
            symbol = identifiers["symbol"]
            #xbrl = XBRL.from_filing(filing)
            xbrl = filing.xbrl()
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
                df = val.to_dataframe().reset_index(drop=True) ##for deeper breakdown save here :)
                df = df.replace("", None)
                df = pl.from_pandas(df)
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
                if stmt_df is None:
                    stmt_df = df_t
                else:
                    cols = set(stmt_df.columns) | set(cols) 
                    stmt_df = stmt_df.join(df_t, how="left", on=self._identifiers).select(cols) 
                
            cols = self._identifiers + self._standard_concepts
            if not set(stmt_df.columns).issubset(set(cols)):
                print(f"[WARNING]Columns of statements ({symbol}) is not a subset of total standard concepts")
            stmt_df = stmt_df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in cols if c not in stmt_df.columns]).select(cols)
            return stmt_df
        except Exception as e:
            print(f"[WARNING]Failed to fetch statements ({symbol}) due to {e}")
            return None    

    def _filing_section(self, identifiers, filing):
        sections = ["business", "risk_factors", "management_discussion"]
        time.sleep(self._sleep_s)
        try:
            symbol = identifiers["symbol"]
            report = filing.obj()
            print(report.chunked_document)
            print(report.document)
            print(report.sections)
            stmt = report.financials
            print(stmt.income_statement()) ##maybe use
            print(stmt.balance_sheet())
            print(stmt.cashflow_statement())
            row = identifiers | {"accession": filing.accession_number}
            for section_name in sections:
                if hasattr(report, section_name):
                    row[section_name] = getattr(report, section_name)
                else:
                    row[section_name] = None
            return row
        except Exception as e:
            print(f"[WARNING]Failed to fetch section ({symbol}) due to {e}")
            return None

    def _fling_text(self, identifiers, filing, filedir):
        filedir = filedir / "texts"
        filedir.mkdir(parents=True, exist_ok=True)
        time.sleep(self._sleep_s)
        try:
            symbol = identifiers["symbol"]
            filepath = filedir / f"{'_'.join(str(v) for v in identifiers.values())}.txt.gz" 
            text = filing.text() 
            with gzip.open(filepath, "wt", encoding="utf-8") as f: #
                f.write(text)
            print(f"Words: {len(text):,}")
            return filepath
        except Exception as e:
            print(f"[WARNING]Failed to fetch text ({symbol}) due to {e}")
            return None

    def _fling_markdown(self, identifiers, filing, filedir):
        filedir = filedir / "markdowns"
        filedir.mkdir(parents=True, exist_ok=True)
        time.sleep(self._sleep_s)
        try:
            symbol = identifiers["symbol"]
            filepath = filedir / f"{'_'.join(str(v) for v in identifiers.values())}.zst" 
            md = filing.markdown() 
            compressed = self._cctx.compress(md)
            with open(filepath", "wb") as f:
                f.write(compressed)
            print(f"Words: {len(md):,}")
            return filepath
        except Exception as e:
            print(f"[WARNING]Failed to fetch markdown ({symbol}) due to {e}")
            return None


    def _fetch_edgar(self, api_key, filings, symbol, func, form, filedir, startDate): 
        print(f"[INFO]Fetching {symbol}: {startDate}") #{EntityFiling: [text, markdown, xbrl, obj]}
        time.sleep(self._sleep_s)

        if func == "statements" and form in ["10-K", "10-Q"]:
            output = None
        elif func == "sections" and form in ["10-K", "20-F"]:
            output = []
        elif func == "texts":
            output = None
        elif func == "markdowns":
            outut=None
        else:
            return None
            
        for filing in filings:
            time.sleep(self._sleep_s)
            identifiers =  {"symbol": symbol, "date": filing.filing_date}
            
            if func == "statements":
                stmt_df = self._filing_stmts(identifiers, filing) #1 HTTP request for xbrl
                output = pl.concat([output, stmt_df], how="vertical") if output is not None else stmt_df
            
            if func == "sections":
                section_row = self._filing_section(identifiers, filing) #1 HTTP request for obj
                if section_row is not None:
                    output.append(section_row)
                
            if func == "texts":
                text_path = self._filing_text(identifiers, filing, filedir) #1 HTTP request for text

            if func == "markdowns":
                markdown_path = self._filing_markdown(identifiers, filing, filedir) #1 HTTP request for markdown
                
        if isinstance(output, list):
            return pl.DataFrame(output)
        return output
    
    def load_sec(self, database, form="10-K", funcs=["statements", "sections", "markdowns"], refresh_threshold_days=None, REFRESH=False):
        partition_cols = self._config["equities"]["partitions"][:self._partition_degree]
        database = database.with_columns([pl.col(c).cast(pl.Utf8).fill_null("Unknown") for c in partition_cols])
        filedir = self._basepath / "exclude" 
        exclude_lazy = self._reader.read_lazy(filedir, identifiers=["symbol"])

        total_calls = len(database)
        failed_calls = []
        max_days_elapsed = 0
        outputs = {elem: None for elem in funcs}
        
        print(f"\n[INFO]Fetching SEC Data ({total_calls})")
        for row in database.iter_rows(named=True):
            startDate = None
            symbol=row["symbol"]
            company = Company(symbol) 
            filings = company.get_filings( #1 HTTP request per symbol (json containing every filing, regardless of date)
                form=form,
                filing_date=f"{self._history_start}:", 
                amendments=False
            )

            func = "statements" #testing
            filedir = self._basepath / "equities" / func / form
            lf = self._reader.read_lazy(filedir, identifiers=self._identifiers)
            print(filedir)
            
           

            #SKIP -> Already Cached
            if not REFRESH: 
                df_old = self._reader.filter_lazy(lf, filters={"symbol": [symbol]}, COLLECT=True) 
                exist_bool = not (df_old is None or df_old.is_empty())
                if exist_bool and "date" in df_old.columns:
                    startDate = df_old["date"].max().strftime("%Y-%m-%d")  
                    days_elapsed = (self._today - df_old["date"].max()).days
                    if (refresh_threshold_days is None or (days_elapsed <= refresh_threshold_days)):
                        max_days_elapsed = max(days_elapsed, max_days_elapsed)
                        continue

            #FETCH -> Symbol
            if startDate is not None:
                filings = filings.filter(filing_date=f"{startDate}:")

            print(filings)
            for source, api_key in self._sources.items(): 
                #Skip -> Already Failed
                if exclude_lazy is not None:
                    IsInExclude = not self._reader.filter_lazy(
                        exclude_lazy,
                        filters={"symbol": [symbol], "asset_type": ["equities"], "func": [func], "form": [form], "source": [source]}, inclusive=True
                        ).limit(1).collect().is_empty()
                    if IsInExclude:
                        continue 

                #FETCH -> Source
                try:
                    if source == "edgar":                        
                        df_new = self._fetch_edgar(api_key, filings, symbol, func, form, filedir, startDate)
                    else:
                        raise ValueError(f"Unknown source '{source}'")
                    if df_new is None:
                        raise ValueError(f"Empty source '{source}'")

                    outputs[func] = pl.concat([outputs[func], df_new]) if outputs[func] is not None else df_new
                    print(f"[INFO]Successfully fetched {symbol}")
                    break 
                except Exception as e:
                    print(f"[WARNING]Error fetching {symbol}: {e}")
                    df_new = {"symbol": symbol, "asset_type": "equities", "func": func, "form": form, "source": source, "error": str(e).split("\n")[0][:100]}
                    failed_calls.append(df_new)
                time.sleep(1.2)

        ##batch write and tar + zst , markdown? (symbol_date.txt.gz)?
        #self._writer.append_parquet(filedir, outputs[func], source=source, identifiers=self._identifiers)
        success_rate = ((total_calls - len(failed_calls)) / total_calls) * 100 if total_calls>0 else np.nan
        print(f"[INFO]Successfully fetched: {success_rate:.2f}% of symbols ({total_calls})")
        print(f"[INFO]Maximum days elapsed: {max_days_elapsed}")

        symbols = list(database["symbol"].unique())
        filedir = self._basepath / "equities" / func / form
        lf = self._reader.read_lazy(filedir, identifiers=self._identifiers) 
        output_df = self._reader.filter_lazy(lf, filters={"symbol": symbols}, COLLECT=True)
        
        failed_df = pl.DataFrame(failed_calls) if failed_calls else None
        filedir = self._basepath / "exclude" 
        self._writer.append_parquet(filedir, failed_df, source=None, identifiers=["symbol"]) 
        return output_df, failed_df

##parse xrbl and text using ml + RAG












