#%%

import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import plotly.express as px
from IPython.display import Image as IImage
from matplotlib import pyplot as plt
from PIL import Image
import io
import math

#%%
#Output:
# The coordinates range is:
#   - Latitude: -36.913779 to 52.900803
#   - Longitude: 31.992111 to 51.037119
# 
# The coordinates range is:
#   - Latitude: -37.0 to 53.0
#   - Longitude: 31.0 to 52.0

# Define the coordinates for two points
#p1 = (-37, 53)
#p2 = (31, 52)
p1 = (41.2752,-8.7637)
p2 = (41.1200,-8.4698)
#l=99.999
#p1 = (42.151023, -6.377139)
#p2 = (38.266623, -9.446562)
#l=99.99
#p1 = (42.151023, -6.676938)
#p2 = (38.661705, -9.378936)

# Calculate the bounding box for the two points
north = max(p1[0], p2[0])
south = min(p1[0], p2[0])
east = max(p1[1], p2[1])
west = min(p1[1], p2[1])

# Create a graph from the bounding box
G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive')
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
ox.plot_graph(G)

#%%

# Calculate the shortest path between the two points
#origin_point        = (-8.644887, 41.159151)
#destination_point   = (-8.687808, 41.163678)
origin_point        = (-8.674110, 41.187840)
destination_point   = (-8.603780, 41.143650)
orig_node = ox.distance.nearest_nodes(G, origin_point[0], origin_point[1])
destination_node = ox.distance.nearest_nodes(G, destination_point[0], destination_point[1])
start_node = G.nodes.get(orig_node)
#end_node = G.nodes.get(destination_node)
route = nx.shortest_path(G, orig_node, destination_node, weight='travel_time')
route_edges = ox.routing.route_to_gdf(G, route, 'travel_time')
total_route_length = route_edges['length'].sum()
route_travel_time = route_edges['travel_time'].sum()
print("Total route length in km:", total_route_length/1000)
print("Travel time in minutes:", route_travel_time/60)


fig, ax = ox.plot_graph_route(G, route, node_size=0, figsize=(40,40))

#%%
route_travel_time / 15

#%%
def get_gps_points_for_route(G, route):
    gps_points = []
    time_passed = 0
    time_interval = 15
    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
        
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
                    print("Error: The fu ??? How??")
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

gps_points = get_gps_points_for_route(G, route)
gps_points.shape

#%%
# Plot gps points

fig, ax = ox.plot.plot_graph(G, show=False, save=False, close=False, node_size=0, figsize=(40,40))
ax.plot(gps_points[:,0], gps_points[:,1], c="r", lw=4, alpha=0.5)
plt.show()

#%%

border_width = (-8.4, -8.8)
border_height = (41.1, 41.3)

center = sum(border_width)/2, sum(border_height)/2

rel_width = abs(border_width[0] - border_width[1])
rel_height = abs(border_height[0] - border_height[1])
aspect = rel_width / rel_height
map_height = 600
map_width = int(aspect * map_height)

map_height, map_width

#%%





#%%

## Test
#import rerun as rr
#
#rr.init("gen_taxi_trajectoriers")
#rr.connect()
#
#rr.log("image", rr.Image(fig_img_np))
#rr.log("bounds", rr.Boxes2D(sizes=[rel_width, rel_height]))
