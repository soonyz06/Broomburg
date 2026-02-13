from pathlib import Path
import polars as pl
import pandas as pd
import time

from src.utils.logg import log_info, log_df
from src.io.sec import EdgarToolsAPI
from src.nlp.bert import SimpleBertChunker




edgar = EdgarToolsAPI(0.5)
history_start = "2023-01-01"

api_keys = load_apikeys()
api_key = api_keys.get("GEMINI_API_KEY", None)
client = genai.Client(api_key=api_key)


symbol = "AAPL"
t0 = log_info("Fetching")
filings = edgar.fetch_filings(symbol, ["10-K", "20-F"], startDate=history_start)
df = edgar.process_obj(symbol, filings, limit=1).sort("date", descending=False)
log_info("Fetching", t0)   

t0 = log_info("Chunking") 
chunker = SimpleBertChunker()
sections = df.tail(1).select(['business', 'risk_factors', 'management_discussion'])
for section in sections.columns:
    chunks = chunker.chunk_text(sections[section][0], threshold=0.7, min_buffer_size=2)
    ##overlap + metadata
log_info("Chunking", t0)

for section in sections:
    print(section.title())
    for chunk in chunks:
        time.sleep(1)
        summary = client.models.generate_content(
            model="models/gemini-2.5-flash", 
            contents=f"Summarise this chunk of text in 1 sentence.: {chunk}",
        )
        print(summary.text)
        print("")
    print("\n")

#vector+keyword search -> Cross-encoder re-ranking -> format for llm




 
"""
response = 

print(response.text)


client = genai.Client(api_key="YOUR_API_KEY_HERE")
chat = client.chats.create(model="gemini-3-flash")

response = chat.send_message("Hi, I'm learning Python.")
print(f"Gemini: {response.text}")

response = chat.send_message("What was the first thing I told you?")
print(f"Gemini: {response.text}")
"""

















