#%%
import ujson as json
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm 
import pyarrow.parquet as pq
#%%


def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")



INPUT_FILE = 'data/synthetic_data.parquet'
OUTPUT_FILE = "data/synthetic_gps_data.parquet"

# Read Data
log("Loading the data...")
synthetic_data = pq.read_table(INPUT_FILE).to_pandas()


log("Load map data...")
geofence_p1 = (41.2752,-8.7637)
geofence_p2 = (41.1200,-8.4698)
G = ox.graph_from_bbox(bbox=(geofence_p1[0], geofence_p2[0], geofence_p1[1], geofence_p2[1]), network_type='drive')
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)


#%%

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_gps_points_for_route(G, route):
    gps_points = []
    time_passed = 0
    time_interval = 15

    # Add the first point
    first_node = G.nodes[route[0]]
    gps_points.append((first_node["x"], first_node["y"]))

    # Iterate over the edges in the route
    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d["travel_time"])
        
        # Check if the travel time is within the time interval
        travel_time = data['travel_time']
        if time_passed + travel_time <= time_interval:
            time_passed += travel_time
            continue

        in_edge_traveled_time = 0
        while in_edge_traveled_time < travel_time:
            missing_to_interval = time_interval - time_passed
            percent_of_edge = (in_edge_traveled_time + missing_to_interval) / travel_time
            if percent_of_edge > 1: 
                time_passed += (time_passed + (travel_time - in_edge_traveled_time)) % time_interval
                break
            time_passed = (time_passed + missing_to_interval) % time_interval
            in_edge_traveled_time += missing_to_interval
            
            # Calculate the gps point
            if "geometry" in data:
                geo_path = np.array(data["geometry"].xy).T
                geo_path_diffs = np.linalg.norm(geo_path[:-1] - geo_path[1:], axis=1)
                geo_path_length = geo_path_diffs.sum()
                point_on_path = percent_of_edge * geo_path_length
                # Select the two points that the point_on_path is between
                checked_length = 0
                gps_point = None
                for i in range(len(geo_path_diffs)):
                    diff = geo_path_diffs[i]
                    if checked_length + diff >= point_on_path:
                        from_point = geo_path[i]
                        to_point = geo_path[i+1]
                        local_percent_diff = (point_on_path - checked_length) / diff
                        gps_point = from_point + (to_point - from_point) * local_percent_diff
                        break
                    checked_length += diff
                if gps_point is None: 
                    #print(f"Error: gps_point is None this should not happen! (edge_id = {data['osmid']}, duration = {travel_time})")
                    #raise nx.exception.NetworkXNoPath()
                    u_node, v_node = G.nodes[u], G.nodes[v]
                    from_point = np.array([u_node["x"], u_node["y"]])
                    to_point = np.array([v_node["x"], v_node["y"]])
                    gps_points.append(from_point + (to_point - from_point) * percent_of_edge)
                    break
                gps_points.append(gps_point)

            else: # Otherwise, the edge is a straight line from node to node
                u_node, v_node = G.nodes[u], G.nodes[v]
                from_point = np.array([u_node["x"], u_node["y"]])
                to_point = np.array([v_node["x"], v_node["y"]])
                gps_points.append(from_point + (to_point - from_point) * percent_of_edge)

    # Add the last point
    x, y = G.nodes[v]["x"], G.nodes[v]["y"]
    gps_points.append((x, y))
    return np.array(gps_points)

def gen_route_from_points(G, origin_point, destination_point):
    orig_node = ox.distance.nearest_nodes(G, *origin_point)
    destination_node = ox.distance.nearest_nodes(G, *destination_point)
    route = nx.shortest_path(G, orig_node, destination_node, weight='travel_time')
    route_edges = ox.routing.route_to_gdf(G, route, 'travel_time')
    total_route_length = route_edges['length'].sum() # in meters
    route_travel_time = route_edges['travel_time'].sum() # in seconds
    return route, total_route_length, route_travel_time
    

def gen_gps_points_from_router_start_and_end_points(G, origin_point, destination_point):
    try:
        route, length, duration = gen_route_from_points(G, origin_point, destination_point)
        gps_points = get_gps_points_for_route(G, route)
        return gps_points, length, duration
    except nx.exception.NetworkXNoPath:
        #print(f"No path found between {origin_point} and {destination_point}")
        return None, None, None

print("Generating the synthetic data with GPS points...")
# Hint: Tried to use joblib.Parallel but it was not working properly with tqdm
pprocess = ProgressParallel(n_jobs=1, use_tqdm=True, total=len(synthetic_data))


row_iterator = synthetic_data[['START_POSITION_LONG', 'START_POSITION_LAT', 'END_POSITION_LONG', 'END_POSITION_LAT']].itertuples(index=False)
data_with_poly = pprocess(delayed(gen_gps_points_from_router_start_and_end_points)(G, row[:2], row[2:]) for row in row_iterator)
df_output = pd.DataFrame(data_with_poly, columns=['GPS_POINTS', 'ROUTE_LENGTH', 'ROUTE_DURATION'])
del data_with_poly

df_output['TRIP_LENGTH'] = df_output['GPS_POINTS'].apply(lambda x: x.shape[0] if x is not None else 0)
df_output

#%%
# # Post-process the data

# Merge the data
for oc, dc in [
    ('TRIP_DURATION', 'ROUTE_DURATION'),
    ('TRIP_LENGTH', 'TRIP_LENGTH'),
    ('ROUTE_LENGTH', 'ROUTE_LENGTH'),
    ('POLYLINE', 'GPS_POINTS')
]:
    synthetic_data[oc] = df_output[dc]

# Drop rows with no GPS points
synthetic_data = synthetic_data[~synthetic_data['POLYLINE'].isnull()]

# Convert the GPS points to string
synthetic_data['POLYLINE'] = synthetic_data['POLYLINE'].apply(lambda x: json.dumps(x.tolist()) if x is not None else None)

# Combine the start and end positions
synthetic_data['START_POSITION'] = synthetic_data[['START_POSITION_LONG', 'START_POSITION_LAT']].apply(lambda x: f"({x['START_POSITION_LONG']} {x['START_POSITION_LAT']})", axis=1)
synthetic_data['END_POSITION']   = synthetic_data[['END_POSITION_LONG', 'END_POSITION_LAT']].apply(lambda x: f"({x['END_POSITION_LONG']} {x['END_POSITION_LAT']})", axis=1)


# Save the data
log("Saving the data...")
out_df = synthetic_data[['TIMESTAMP', 'TRIP_DURATION', 'TRIP_LENGTH', 'ROUTE_LENGTH', 'START_POSITION', 'END_POSITION', 'POLYLINE']]
synthetic_data.to_parquet(OUTPUT_FILE)
synthetic_data.to_csv('data/synthetic_gps_data.csv', sep=';', index=False, header=True)
log("Done!")

#%%