from edgar import set_identity, Company, get_current_filings, use_local_storage, download_filings
from edgar.xbrl import XBRL, XBRLS
import time
import datetime
from pathlib import Path

from src.utils.logg import log_info, df_display
from src.io.fetch import get_q_dates


t0 = log_info("Fetch")
basepath = str(Path.cwd() / "data" / "raw" / "sec")
set_identity("soon.yz061025@gmail.com")
use_local_storage()

forms = {
    "annual": ["10-K"],
    "quarter": ["10-Q"],
    "material": ["8-K"],
    "insider": ["3", "4", "5"],
    "fund": ["13F-HR"],
    "ipo": ["S-1"]
}

"""
company = Company("AAPL")
latest_filings = get_current_filings(form=forms) #math.ceil(100/page_size) HTTP requests 
latest_filings = get_filings(form=forms).latest(100) #1 HTTP request per filing
filings = company.get_filings( #1 HTTP request per ticker (json containing every filing)
    filing_date="2024-01-01:",
    form = forms["annual"],
    amendments=False
)
#Filings only include metadata (indexes), additional HTTP requests are made when they are accessed
"""

"""
symbols = ["AAPL", "META", "NVDA"]
all_filings = []
for symbol in symbols:
    company = Company(symbol)
    filings = company.get_filings( 
        filing_date="2024-01-01:",
        form = forms["annual"],
        amendments=False
    )
    all_filings.extend(filings)

download_filings(all_filings, basepath)

for filing in all_filings:
    #[Stuff Here]
"""

"""
if filings:
    if not filings[0].has_xbrl():
        print("Filing does not contain XBRL data")
        break
    
    xbrls = XBRLS.from_filings(filings)
    stmts = xbrls.statements
    print(dir(stmts))
   
    IS = stmts.income_statement()
    BS = stmts.balance_sheet()
    CF = stmts.cashflow_statement()
    ES = stmts.statement_of_equity()
    CI = stmts.comprehensive_income()

    df = IS.to_dataframe()
    df = df.set_index(df.columns[0]).T
    print(df.head())
    print(df.columns)
    """

#company.industry
for filing in filings:
    print(f"{filing.form} ({filing.file_number}): {filing.company} ({filing.filing_date})")
    #print(filing.sections())
    #print(filing.text)

    """
    report = filing.obj()
    sections = ["business", "risk_factors", "management_discussion"]
    for section_name in sections:
        if hasattr(report, section_name):
            section = getattr(report, section_name)
            print(section)
            del section
    time.sleep(0.2)
    """
    break
    

def monitor_current_events(forms, n=100):
    current = get_current_filings(form=forms[0], page_size=n)
    print(f"📈 Monitoring {len(current)} recent {forms} filings")
    print(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    for filing in current:
        print(f"{filing.company}")
        print(f"  Form: {filing.form}")
        print(f"  Filed: {filing.filing_date}")
        print(f"  URL: {filing.filing_url}")
        print(filing.summary())
        print()
        time.sleep(0.2)

monitor_current_events(forms["material"])
  
#KeyboardInterrupt Ctrl+C
"""
except CompanyNotFoundError:
    print("[WARNING]Company not found")
except ValueError as e:
    print(f"Invalid identifier: {e}")
"""

"""
@functools.lru_cache(maxsize=128)
def process_filings_generator(filings):
    for filing in filings:
        # Process one filing at a time
        result = process_filing(filing)
        yield result
        # Free memory
        del filing

# Process filings one at a time
for result in process_filings_generator(all_filings):
    save_or_analyze(result)

index_data.append({
        "accession": filing.accession_number,
        "cik": filing.cik,
        "company": filing.company_name,
        "form": filing.form_type,
        "date": filing.filing_date,
        "path": filing.get_local_path() if filing.is_local() else None
    })

startDate = maxDate
"""

