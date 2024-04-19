
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import ujson as json
import numpy as np
import matplotlib.pyplot as plt
import haversine as hs


MIN_TRIP_DURATION = 60 * 2 # seconds
MAX_TRIP_DURATION = 60 * 60 * 2 # seconds
MIN_TRIP_DISTANCE = 1000 # meters
MAX_DISTANCE_BETWEEN_POINTS = 1000 # meters
INPUT_FILE = 'data/train.parquet'
OUTPUT_FILE = "data/train_cleaned.parquet"


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km * 1000

def remove_outliers(df, min_duration=MIN_TRIP_DURATION, max_duration=MAX_TRIP_DURATION, min_distance=MIN_TRIP_DISTANCE, max_distance_between_points=MAX_DISTANCE_BETWEEN_POINTS):
    """
    Remove some outliers that could otherwise undermine the training's results.
    """
    # Remove trips that are either extremely long or short (potentially due to GPS recording issue)
    indices = np.where((df["TRIP_DURATION"] > min_duration) & (df["TRIP_DURATION"] <= max_duration))
    df = df.iloc[indices]

    # Remove trips that are too far away from Porto (also likely due to GPS issues)
    bounds = (  # Bounds retrieved using http://boundingbox.klokantech.com
        (41.052431, -8.727951),
        (41.257678, -8.456039),
    )
    indices = np.where(
        (bounds[0][0] <= df["MIN_LAT"]) & (df["MAX_LAT"] <= bounds[1][0]) &
        (bounds[0][1] <= df["MIN_LONG"]) & (df["MAX_LONG"] <= bounds[1][1])
    )
    df = df.iloc[indices]

    # Remove trips that are too short (likely due to GPS issues)
    indices = np.where(df["TRIP_DISTANCE"] > min_distance)
    df = df.iloc[indices]

    # Remove trips that have two consecutive points that are too far away from each other
    indices = np.where(df["MAX_DISTANCE_BETWEEN_POINTS"] < max_distance_between_points)
    df = df.iloc[indices]

    return df


df = pq.read_table(INPUT_FILE).to_pandas()

# Convert the POLYLINE column to a list of coordinates
df['POLYLINE'] = df['POLYLINE'].apply(lambda x: np.array(json.loads(x)))
df['TRIP_LENGTH'] = df['POLYLINE'].apply(lambda x: x.shape[0])
df = df[df['TRIP_LENGTH'] > 3] # Remove trips with less than 4 coordinates
df['POLYLINE'] = df['POLYLINE'].apply(lambda x: x[:, [1, 0]])

df["TRIP_DURATION"] = df["TRIP_LENGTH"] * 15
tmp = df["POLYLINE"].apply(lambda x: haversine_np(x[:-1, 0], x[:-1, 1], x[1:, 0], x[1:, 1]))
df["TRIP_DISTANCE"] = tmp.apply(lambda x: np.sum(x))
df["MAX_DISTANCE_BETWEEN_POINTS"] = tmp.apply(lambda x: np.max(x))
df = df[df["TRIP_DISTANCE"] > 0] # Remove trips with 0 distance

df["MIN_LAT"] = df["POLYLINE"].apply(lambda x: x.min(axis=0)[0])
df["MAX_LAT"] = df["POLYLINE"].apply(lambda x: x.max(axis=0)[0])
df["MIN_LONG"] = df["POLYLINE"].apply(lambda x: x.min(axis=0)[1])
df["MAX_LONG"] = df["POLYLINE"].apply(lambda x: x.max(axis=0)[1])
df = remove_outliers(df)

df["START_POSITION_LAT"] = df["POLYLINE"].apply(lambda x: x[0][0])
df["START_POSITION_LONG"] = df["POLYLINE"].apply(lambda x: x[0][1])
df["END_POSITION_LAT"] = df["POLYLINE"].apply(lambda x: x[-1][0])
df["END_POSITION_LONG"] = df["POLYLINE"].apply(lambda x: x[-1][1])

df = df[[
    'TRIP_ID', 'TIMESTAMP', 'TRIP_DURATION', 
    'TRIP_LENGTH', 'TRIP_DISTANCE', 'POLYLINE',
    'START_POSITION_LAT', 'START_POSITION_LONG', 
    'END_POSITION_LAT', 'END_POSITION_LONG',
]]
df = df.reset_index(drop=True)

# Export data
df["POLYLINE"] = df["POLYLINE"].apply(lambda x: json.dumps(x.tolist()))
df.to_csv("data/train_cleaned.csv", index=False)
pq.write_table(pa.Table.from_pandas(df), OUTPUT_FILE)
