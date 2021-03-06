This is a graph theory application of a road route planner in Python and a visualization 
of the route using an OpenStreetMap.  

To run this tool simply download the data (a Shapefile dataset) and the route_planner.py file.  

In route_planner.py, input your starting and ending positons as a latitutde/lognitude coordinates and run the file to get a map of your route. Note - The current code also has a break point along the route. With a few alteratoins to the code, a route between any number of points/destinations can be created. 

Furthermore the roads pandas dataframe is a cue sheet of the routes specific roads and the total distance of the route is given by 
```python
roads['distance'].sum()
```

This routing tool implements Dijkstra's algorithm to find the shortest path between the given starting and ending points. 

More info and visuals can be found here at http://www.davidbangor.com/blog relating this routing tool to a bicycle tour I completed in the summer of 2007.