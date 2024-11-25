import os
import sys
import numpy as np

x = np.array([[1,2], [3,4], [5,6]])

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)


from solutions.my_drone_random import MyDroneRandom
import math

def shortest_path(start , end, grid) :
    start = grid._conv_world_to_grid(*start)

    value_boxes = {(i, j) : float('inf') 
                   for j in range(len(grid.grid[0])) 
                   for i in range(len(grid.grid)) 
                   if grid.grid[i, j] < -0.6}
    
    value_boxes[start] = 0
    
    current = start
    x, y = current
    closest = {node : None for node in value_boxes}
    del value_boxes[current]
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    n = 0

    while value_boxes :
        if current == end :
            break

        for dx, dy in directions :
            if x + dx in range(len(grid.grid)) and y + dy in range(len(grid.grid[0])) :
                d = math.sqrt(2) if 0 not in (dx, dy) else 1
                if (x + dx, y + dy) in value_boxes and value_boxes[(x + dx, y + dy)] > n + d :
                    value_boxes[(x + dx, y + dy)] = n + d
                    closest[(x+dx, y+dy)] = current
                    
        x, y = current = min(value_boxes, key = value_boxes.get)
        n = value_boxes[current]
        del value_boxes[current]

    p = [end]
    current = end
    while current != start :
        p.append(closest[current])
        current = closest[current]

    return p[::-1]
