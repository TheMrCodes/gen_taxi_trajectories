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

#Output:
# The coordinates range is:
#   - Latitude: -36.913779 to 52.900803
#   - Longitude: 31.992111 to 51.037119
# 
# The coordinates range is:
#   - Latitude: -37.0 to 53.0
#   - Longitude: 31.0 to 52.0

#%%
# calculate area size
area_size = (all_max - all_min)
print(f"The area size is: {area_size[0]} x {area_size[1]} = {area_size[0] * area_size[1]} degree^2")

#%%
# calculate individual nodes
#   a node is a unique combination of latitude and longitude

all_points = np.concatenate(df_tray['POLYLINE'].to_numpy()).reshape(-1, 2)
print(f"Total number of points: {len(all_points)}")
all_points = np.unique(all_points, axis=0)
print(f"Total number of unique points: {len(all_points)}")

#%%
all_points2 = np.concatenate(df_tray['POLYLINE'].to_numpy()).reshape(-1, 2)
point_cnt = pd.DataFrame(all_points2, columns=['lat', 'lon']).groupby(['lat', 'lon']).size().reset_index(name='count')
point_cnt = point_cnt.sort_values('count', ascending=False)
point_cnt['count'].hist(bins=100)

#%%
## print headmap of the values in point_cnt x = lat, y = lon, z = count, x-range = min-max lat, y-range = min-max lon
#sns.heatmap(
#    data=point_cnt, annot=True, fmt='d',
#    xticklabels=10, yticklabels=10,
#)

#%%

DATA_FILE = 'data/train.parquet'

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import ujson as json
import numpy as np
import matplotlib.pyplot as plt

df = pq.read_table(DATA_FILE).to_pandas()
df['POLYLINE'] = df['POLYLINE'].apply(lambda x: np.array(json.loads(x)))
df['POLYLINE_LEN'] = df['POLYLINE'].apply(lambda x: len(x))
df = df[df['POLYLINE_LEN'] > 0]
points = np.concatenate(df['POLYLINE'].to_numpy()).reshape(-1, 2)

#%%


# draw scatter plot of the points
import matplotlib.pyplot as plt
plt.scatter(points[:, 0], points[:, 1], s=1)
plt.show()

#%%

import numpy as np
from scipy.stats import chi2

# calculate the covariance matrix of the points
cov = np.cov(points, rowvar=False)
# calculate the eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov)

# sort the eigenvalues and eigenvectors in descending order
eig_vals_sorted = np.sort(eig_vals)[::-1]
eig_vecs_sorted = eig_vecs[:, np.argsort(eig_vals)[::-1]]

# calculate the ellipse parameters (center, width, height, angle)
center = np.mean(points, axis=0)
width = 2 * np.sqrt(eig_vals_sorted[0] * chi2.ppf(0.95, 2))  # 95% confidence interval
height = 2 * np.sqrt(eig_vals_sorted[1] * chi2.ppf(0.95, 2))  # 95% confidence interval
angle = np.arctan2(eig_vecs_sorted[1, 1], eig_vecs_sorted[0, 1])

# function to check if a point is inside the ellipse
def is_inside_ellipse(point, center, width, height, angle):
    c, s = np.cos(angle), np.sin(angle)
    Rx = np.array([[c, -s], [s, c]])
    point_rotated = np.dot(Rx, point - center)
    return (point_rotated[0] / width) ** 2 + (point_rotated[1] / height) ** 2 <= 1

# filter out points outside the ellipse
filtered_points = [point for point in points if is_inside_ellipse(point, center, width, height, angle)]

#%%

threshold_percentile = 99.99

# Calc z-scores
mean_x, mean_y = np.mean(points, axis=0)
std_x, std_y = np.std(points, axis=0)
z_scores_x = np.abs((points[:, 0] - mean_x) / std_x)
z_scores_y = np.abs((points[:, 1] - mean_y) / std_y)
z_scores = np.sqrt(z_scores_x**2 + z_scores_y**2)
threshold = np.percentile(z_scores, threshold_percentile)

filtered_points = points[z_scores <= threshold]

# plot the points and the ellipse
plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1)
plt.axis('equal')
plt.show()

#%%
mean_x, mean_y, std_x, std_y

#%%
(*filtered_points.max(axis=0), *filtered_points.min(axis=0))