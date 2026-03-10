# model

# Data Pipeline
* Tabular data stored as parquet files using [PM](src/io/parquet.py):
  - Data is parititioned, where each subtable is broken down into batches which are each written to a new file with constraints on max row group and file sizes
  - Write to new file each time + Scheduled Compact job 
  - 
