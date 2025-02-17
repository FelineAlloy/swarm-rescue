import os
import sys
import numpy as np
from queue import PriorityQueue

x = np.array([[1,2], [3,4], [5,6]])

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)


from solutions.my_drone_random import MyDroneRandom
import math

def shortest_path(start, end, grid) :
    end = (int(end[0]), int(end[1])) # Convert np.array to tuple

    cost = np.full((grid.x_max_grid, grid.y_max_grid), np.inf)
    pq = PriorityQueue() # (dist, (x, y))
    viz = np.full((grid.x_max_grid, grid.y_max_grid), False) 
    closest = dict() # (x, y): (x1, y1)
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    pq.put((0, start))
    cost[start] = 0

    while not pq.empty():
        x, y = current = pq.get()[1]
        dist = cost[current]
        viz[current] = True

        for dx, dy in directions :
            if 0 <= x + dx < grid.x_max_grid and 0 <= y + dy < grid.y_max_grid and grid.grid[x+dx, y+dy] < -0.6:
                d = math.sqrt(2) if 0 not in (dx, dy) else 1
                if dx * dy != 0:
                    if not (grid.grid[x, y+dy] < -0.6 and grid.grid[x+dx, y] < -0.6):
                        continue
                if not viz[(x + dx, y + dy)] and cost[(x + dx, y + dy)] > dist + d :
                    cost[(x + dx, y + dy)] = dist + d
                    closest[(x+dx, y+dy)] = current
                    pq.put((dist + d, (x + dx, y + dy)))

    if end not in closest: # this is just a temporary patch
        print(f"Warning: No path found from {start} to {end}")
        return [start]

    p = [end]
    current = end
    while current != start :
        p.append(closest[current])
        current = closest[current]

    return p[::-1]
