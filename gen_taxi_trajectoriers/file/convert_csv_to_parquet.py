# File: convert_csv_to_parquet.py

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


INPUT_FILE = 'data/train.csv'
OUTPUT_FILE = 'data/train.parquet'


df = pd.read_csv(INPUT_FILE)
table = pa.Table.from_pandas(df)
pq.write_table(table, OUTPUT_FILE)

