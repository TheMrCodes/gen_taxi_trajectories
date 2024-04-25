#%%
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import ujson as json
import numpy as np
import torch
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.linear_model import LinearRegression

# Import Intel extension for pytorch for Intel GPUs
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass
def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif is_xpu_available():
        return "xpu"
    return "cpu"



DATA_FILE = 'data/train_cleaned.parquet'
OUTPUT_FILE = 'data/synthetic_data.parquet'
DEVICE = get_device()

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")


# Read Data
log("Loading the data...")
df = pq.read_table(DATA_FILE).to_pandas()
#df['POLYLINE'] = df['POLYLINE'].apply(lambda x: np.array(json.loads(x)))

#%%

# Names of the columns that are discrete
discrete_columns = [
    'TIMESTAMP', 
    'TRIP_DURATION',
    'TRIP_LENGTH', 
    'START_POSITION_LAT', 
    'START_POSITION_LONG', 
    'END_POSITION_LAT',
    'END_POSITION_LONG',
]
dataset = df[discrete_columns].sample(5000)

ctgan = CTGAN(epochs=10, cuda=DEVICE)
log("Start fitting the model...")
ctgan.fit(dataset, discrete_columns)

# Create synthetic data
log("Start generating sampling...")
synthetic_data = ctgan.sample(1000)

#%%
synthetic_data.to_parquet(OUTPUT_FILE)
#synthetic_data.to_csv('data/synthetic_data.csv', sep=';', index=False, header=True)