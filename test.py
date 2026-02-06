from pathlib import Path
import polars as pl
import pandas as pd

from src.utils.logg import log_info, log_df
from src.io.fetch import load_apikeys
from src.io.sec import EdgarToolsAPI
from src.nlp.bert import SimpleBertChunker

t0 = log_info("Fetching")
edgar = EdgarToolsAPI(0.5)
history_start = "2023-01-01"
    
symbol = "AAPL"
filings = edgar.fetch_filings(symbol, ["10-K", "20-F"], startDate=history_start)
df = edgar.process_obj(symbol, filings, limit=1).sort("date", descending=False)
log_info("Fetching", t0) 
   

t0 = log_info("Chunking") 
chunker = SimpleBertChunker()
sections = df.tail(1).select(['business', 'risk_factors', 'management_discussion'])
for section in sections.columns:
    chunks = chunker.chunk_text(sections[section][0], threshold=0.7, min_buffer_size=2)
    chunker.print_chunk(chunks)
    ##overlap + metadata
log_info("Chunking", t0) 



#vector+keyword search -> Cross-encoder re-ranking -> format for llm


