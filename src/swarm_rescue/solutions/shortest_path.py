import os
import sys
import numpy as np
from queue import PriorityQueue
from collections import deque

x = np.array([[1,2], [3,4], [5,6]])

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)


from solutions.my_drone_random import MyDroneRandom
import math

def shortest_path(start, end, grid, threshold=-0.6):
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
            if 0 <= x + dx < grid.x_max_grid and 0 <= y + dy < grid.y_max_grid and grid.grid[x+dx, y+dy] < threshold:
                d = math.sqrt(2) if 0 not in (dx, dy) else 1
                if dx * dy != 0:
                    if not (grid.grid[x, y+dy] < threshold and grid.grid[x+dx, y] < threshold):
                        continue
                if not viz[(x + dx, y + dy)] and cost[(x + dx, y + dy)] > dist + d :
                    cost[(x + dx, y + dy)] = dist + d
                    closest[(x+dx, y+dy)] = current
                    pq.put((dist + d, (x + dx, y + dy)))

    if end not in closest: # this is just a temporary patch
        cell = lee_fill(grid.grid, end, closest)
        if cell == -1:
            # print(f"Warning: No path found from {start}:{grid.grid[start]} to {end}:{grid.grid[end]}")
            return ([start], end)
        
        # print(f"Warning: found path to alternate cell {cell}, instead of {end}")
        end = cell
        

    p = [end]
    current = end
    while current != start :
        p.append(closest[current])
        current = closest[current]

    return (p[::-1], end)

def lee_fill(grid: np.ndarray, start: tuple, reacheable_cells: dict):
    """
    Performs Lee fill (BFS) on a NumPy 2D array until it finds the closest cell.
    
    Parameters:
    - grid: np.ndarray -> 2D grid representing the space
    - start: tuple -> (row, col) start position
    - closest_cell: tuple -> (row, col) target position to stop at
    
    Returns:
    - path_length: int -> Distance from start to closest_cell
    """
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    # Queue stores (row, col, distance)
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    visited.add(start)
    
    while queue:
        r, c, dist = queue.popleft()
        
        # If we reach a reachable cell, return distance
        if (r, c) in reacheable_cells:
            return (r, c)
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                queue.append((nr, nc, dist + 1))
                visited.add((nr, nc))
    
    return -1  # Return -1 if the closest cell is unreachable