#%%

import osmnx as ox
import networkx as nx


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
#%%
len(G.nodes)
#%%

