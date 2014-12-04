import networkx as nx
import numpy as np
import pandas as pd
import json
import smopy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = mpl.rcParams['savefig.dpi'] = 300

# load in the Shapefile dataset with NetworkX
# returns a graph with each node is a geographical location and
# each edge is info about the road linknig the two nodes
g = nx.read_shp("data/tl_2013_06_prisecroads.shp")

# This graph is not necessarliy connected, so we take the largest connected subgraph
# by using the connected_component_subgraphs function.
# Unfortunately this dataset is not as robust as I had expected.
# The full graph is about 7200 nodes while the lagest connected subgraph 
# is only about 400 nodes, thus limiting the amount of meaningful routes.
sg = list(nx.connected_component_subgraphs(g.to_undirected()))[0]

# Starting => Monterey area
pos0 = (36.6026, -121.9026)

# Taking a break => Los Angeles area
pos1 = (34.0569, -118.2427)

# End => California/Mexico Boarder
pos2 = (32.542181, -117.029543)

def get_path(n0, n1):
    """If n0 and n1 are connected nodes in the graph, this function
    return an array of point coordinates along the road linking
    these two nodes."""
    return np.array(json.loads(sg[n0][n1]['Json'])['coordinates'])

EARTH_R = 6372.8
def geocalc(lat0, lon0, lat1, lon1):
    """Return the distance (in km) between two points in 
    geographical coordinates."""
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    dlon = lon0 - lon1
    y = np.sqrt((np.cos(lat1) * np.sin(dlon)) ** 2 + (np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
    x = np.sin(lat0) * np.sin(lat1) + np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
    c = np.arctan2(y, x)
    return EARTH_R * c

def get_path_length(path):
    return np.sum(geocalc(path[1:,0], path[1:,1], path[:-1,0], path[:-1,1]))

# Compute the length of the road segments.
for n0, n1 in sg.edges_iter():
    path = get_path(n0, n1)
    distance = get_path_length(path)
    sg.edge[n0][n1]['distance'] = distance

nodes = np.array(sg.nodes())

# Get the closest nodes in the graph looking at the Euclidean distance
pos0_i = np.argmin(np.sum((nodes[:,::-1] - pos0)**2, axis=1))
pos1_i = np.argmin(np.sum((nodes[:,::-1] - pos1)**2, axis=1))
pos2_i = np.argmin(np.sum((nodes[:,::-1] - pos2)**2, axis=1))

# Compute the shortest path. Dijkstra's algorithm.
path1 = nx.shortest_path(sg, source=tuple(nodes[pos0_i]), target=tuple(nodes[pos1_i]), weight='distance')
path2 = nx.shortest_path(sg, source=tuple(nodes[pos1_i]), target=tuple(nodes[pos2_i]), weight='distance')

# Get a que sheet of the roads
roads1 = pd.DataFrame([sg.edge[path1[i]][path1[i + 1]] for i in range(len(path1) - 1)], columns=['FULLNAME', 'MTFCC', 'RTTYP', 'distance'])
roads2 = pd.DataFrame([sg.edge[path2[i]][path2[i + 1]] for i in range(len(path2) - 1)], columns=['FULLNAME', 'MTFCC', 'RTTYP', 'distance'])

roads = roads1.append(roads2)

# Call the map
map = smopy.Map(pos0, pos2, z=9, margin=.1)

def get_full_path(path):
    """Return the positions along a path."""
    p_list = []
    curp = None
    for i in range(len(path)-1):
        p = get_path(path[i], path[i+1])
        if curp is None:
            curp = p
        if np.sum((p[0]-curp)**2) > np.sum((p[-1]-curp)**2):
            p = p[::-1,:]
        p_list.append(p)
        curp = p[-1]
    return np.vstack(p_list)

# Get the path
linepath1 = get_full_path(path1)
linepath2 = get_full_path(path2)

x, y = map.to_pixels(linepath1[:,1], linepath1[:,0])
h, k = map.to_pixels(linepath2[:,1], linepath2[:,0])


plt.figure(figsize=(5,8));
map.show_mpl();
# Plot the itinerary.
plt.plot(x, y, '-k', lw=1.5);
plt.plot(h, k, '-k', lw=1.5);
# Mark our two positions.
plt.plot(x[0], y[0], 'ob', ms=5);
plt.plot(h[0], k[0], 'og', ms=5);
plt.plot(h[-1], k[-1], 'or', ms=5);
