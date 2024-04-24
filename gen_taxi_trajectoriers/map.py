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
#origin_point        = (41.159151, -8.644887)
#destination_point   = (41.163678, -8.687808)
origin_point        = (41.187840, -8.674110)
destination_point   = (41.143650, -8.603780)
orig_node = ox.distance.nearest_nodes(G, origin_point[0], origin_point[1])
destination_node = ox.distance.nearest_nodes(G, destination_point[0], destination_point[1])
start_node = G.nodes.get(orig_node)
#end_node = G.nodes.get(destination_node)
route = nx.shortest_path(G, orig_node, destination_node, weight='travel_time')
#edge_lengths = ox.routing.route_to_gdf(G, route, 'length')
#total_route_length = sum(edge_lengths)
#edge_travel_time = ox.routing.route_to_gdf(G, route, 'travel_time')
#route_travel_time = sum(edge_travel_time)
#print("Total route length in km:", total_route_length/1000)
#print("Travel time in minutes:", route_travel_time/60)


fig, ax = ox.plot_graph_route(G, route, node_size=0, figsize=(40,40))
#%%


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

## Test
#import rerun as rr
#
#rr.init("gen_taxi_trajectoriers")
#rr.connect()
#
#rr.log("image", rr.Image(fig_img_np))
#rr.log("bounds", rr.Boxes2D(sizes=[rel_width, rel_height]))
