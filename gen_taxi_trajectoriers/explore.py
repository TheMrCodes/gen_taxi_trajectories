#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

#%%

df = pd.read_csv('data/train.csv')
df

#%%
# Data prep
df['POLYLINE'] = df['POLYLINE'].apply(lambda x: np.array(json.loads(x)))
df['POLYLINE_LEN'] = df['POLYLINE'].apply(lambda x: len(x))

#%%
# Display for exploration
df['POLYLINE_LEN'].hist(bins=100)
df['POLYLINE_LEN'].value_counts()

#%%
df_tray = df[df['POLYLINE_LEN'] > 0][['TRIP_ID', 'POLYLINE']].copy()

#%%
df_tray['POLYLINE_MAX'] = df_tray['POLYLINE'].apply(lambda x: x.max(axis=0))
df_tray['POLYLINE_MIN'] = df_tray['POLYLINE'].apply(lambda x: x.min(axis=0))

#%%
all_max = np.concatenate(df_tray['POLYLINE_MAX'].to_numpy()).reshape(-1, 2).max(axis=0)
all_min = np.concatenate(df_tray['POLYLINE_MIN'].to_numpy()).reshape(-1, 2).min(axis=0)

print(f"The coordinates range is:")
print(f"  - Latitude: {all_min[0]} to {all_max[0]}")
print(f"  - Longitude: {all_min[1]} to {all_max[1]}")
print()

# Round up the values so that min -0.5 becomes -1 and max 0.5 becomes 1
all_min = np.floor(all_min)
all_max = np.ceil(all_max)
print(f"The coordinates range is:")
print(f"  - Latitude: {all_min[0]} to {all_max[0]}")
print(f"  - Longitude: {all_min[1]} to {all_max[1]}")

#%%
# calculate area size
area_size = (all_max - all_min)
print(f"The area size is: {area_size[0]} x {area_size[1]} = {area_size[0] * area_size[1]} degree^2")

#%%
df_poly = df[['POLYLINE']].copy()
df_poly['POLYLINE_LEN'] = df_poly['POLYLINE'].apply(lambda x: len(x))

df_poly['POLYLINE_LEN'].hist()


#%%

#TODO calculate individual nodes
# a node is a unique combination of latitude and longitude