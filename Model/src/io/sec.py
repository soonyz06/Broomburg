from pathlib import Path
import polars as pl
import time
import datetime
import numpy as np
import tarfile
import zstandard as zstd
from edgar import set_identity, Company, get_current_filings


class EdgarToolsAPI:
    def __init__(self, sleep_s):
        #https://edgartools.readthedocs.io/en/latest/api/filing/?h=filing#filingcik-company-form-filing_date-accession_no
        set_identity("soon.yz061025@gmail.com")
        self._sleep_s = sleep_s
        self.forms =  {
            "annual": ["10-K", "20-F"], 
            "quarterly": ["10-Q", "6-K"],
            "current": ["8-K"],
            "insiders": ["3", "4", "5"],
            "ownership": ["13D", "13F", "13G"],
            "registration": ["S-1"],
            "proxy": ["DEF 14A"]
            }
        
    def fetch_filings(self, symbol, forms, startDate):
        print(f"[INFO]Fetching {symbol}: {startDate}")
        time.sleep(self._sleep_s)
        company = Company(symbol) 
        filings = company.get_filings( #1 HTTP request per symbol (json containing every filing, regardless of date)
            form=forms,
            filing_date=f"{startDate}:", 
            amendments=False
        ) 
        return filings #pseudo symbol
    
    def open_filing(self, filings, forms):
        time.sleep(self._sleep_s)
        latest_filing = filings.latest().open()
        return self

    def process_obj(self, symbol, filings, limit):
        if limit<1: return
        filings = filings.latest(limit)
        if limit==1: filings = [filings]
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
                output.append(section_row)
            except Exception as e:
                print(f"[WARNING]Failed to fetch section ({_key}) due to {e}")
        
        if output:
            output = pl.DataFrame(output)
            return output
        return None

    def process_content(self, symbol, filings, func, tardir): 
        mode = func[:-1]
        assert isinstance(mode, str), "Mode should be a string"
        if len(list(filings))>1:
            filings = filings.latest(self._content_limit)

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
