import sys
from pathlib import Path
from typing import Type

import cv2
import numpy as np
import math
from collections import deque

from spg_overlay.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR

from examples.example_mapping import MyDroneMapping

# temp, will have to sync with mapping
EVERY_N = 3
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 2.0
FREE_ZONE_VALUE = -4.0
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40

def count_new_cells(start, grid):
    """
    Counts the number of new cells that will be discovered if the drone moves to start.

    Parameters:
    - start: Tuple (x, y), starting point in the grid.
    - grid: 2D list of numeric values representing the grid.

    Returns:
    - Count of cells meeting the criteria.
    """
    rows, cols = len(grid), len(grid[0])
    start_x, start_y = start
    
    # Directions for 4-connected neighbors
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Queue for BFS
    queue = deque([(start_x, start_y)])  # (x, y)
    visited = set([(start_x, start_y)])  # Track visited cells
    
    count = 0
    
    while queue:
        x, y = queue.popleft()
        
        # Calculate Euclidean distance from the start
        dist = math.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
        
        # Check distance constraint
        if dist > LIDAR_DIST_CLIP:
            continue
        
        # Check value constraint
        if grid[x][y] > OBSTACLE_ZONE_VALUE*3:
            count += 1
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds and if already visited
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return count

def direct_line(p1, p2, grid):
    """
    Verifies if there is a direct line of sight between two points in a grid.

    Parameters:
    - p1: Tuple (x1, y1), starting point.
    - p2: Tuple (x2, y2), ending point.
    - grid: 2D list of numeric values representing the grid.

    Returns:
    - True if there is a direct line of sight, False otherwise.
    """

    x1, y1 = p1
    x2, y2 = p2
    
    # Bresenham's Line Algorithm (simplified to iterate over the line)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    # Initial decision parameter
    err = dx - dy

    while True:
        # Check if the current cell is an obstacle
        if grid[x1][y1] >= OBSTACLE_ZONE_VALUE*3:
            return False

        # Break when we reach the end point
        if x1 == x2 and y1 == y2:
            break

        # Update error term and move to the next cell
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx  # Move horizontally

        if e2 < dx:
            err += dx
            y1 += sy  # Move vertically

    return True

def get_score(drone, cell):
    ''' get the cost of moving from curent position to a cell, 
    taking into account orientation, distance and obstacles.
    '''

    pos = drone.grid._conv_world_to_grid(*drone.estimated_pose.position)
    rot = drone.estimated_pose.orientation

    if direct_line(pos, cell, drone.grid.grid):
        angle_to_cell = math.atan2(cell[1] - pos[1], cell[0] - pos[0])
        return math.sqrt((pos[0] - cell[0])**2 + (pos[1] - cell[1])**2) * math.cos(angle_to_cell - rot)
    
    return count_new_cells(cell, drone.grid.grid)

def count_zeros_around_point(grid, point):
    x, y = point
    rows, cols = grid.shape
    
    # Define the size of the neighborhood (5x5 area)
    size = 5
    count = 0
    
    # Get the bounds of the 5x5 area, ensuring they don't go out of bounds of the grid
    for i in range(x - 2, x + 3):
        for j in range(y - 2, y + 3):
            # Check if the indices are within the bounds of the grid
            if 0 <= i < rows and 0 <= j < cols:
                if grid[i, j] == 0:
                    count += 1
    
    return count

def find_cells(grid):
    """
    Returns a list of cells in the grid that have a value less than 3 * FREE_ZONE_VALUE
    and have at least one neighbor with a value of 0.

    Parameters:
    - grid: 2D NumPy array representing the grid.
    - FREE_ZONE_VALUE: The threshold value to compare against.

    Returns:
    - List of tuples where each tuple represents the coordinates (x, y) of a valid cell.
    """

    # Define the threshold (3 * FREE_ZONE_VALUE)
    threshold = FREE_ZONE_VALUE * 0

    # Create a boolean mask where grid values are less than the threshold
    mask = grid < threshold

    # Get the coordinates of cells that meet the threshold condition
    valid_cells = np.argwhere(mask)

    # Directions for 4-connected neighbors (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    result = []

    # Iterate through valid cells
    for x, y in valid_cells:
        if count_zeros_around_point(grid, (x, y)) > 10:
            result.append((x, y))  # Append (x, y) as (col, row)

    return result

def next_waypoint(drone, boundry_list):
        scores = [(get_score(drone, cell), cell) for cell in boundry_list]
        scores.sort(key=lambda x: x[0])

        if len(scores) > 0:
            return scores[0][1]
        return None

class MyDroneEval(MyDroneMapping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def control(self):
        """
        We only send a command to do nothing
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        # increment the iteration counter
        self.iteration += 1

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        # self.estimated_pose = Pose(np.asarray(self.true_position()),
        #                            self.true_angle())

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0 and self.iteration > 10:

            # boundry_list = find_cells(cv2.resize(self.grid.grid, (self.grid.x_max_grid, self.grid.y_max_grid), interpolation=cv2.INTER_NEAREST))
            # boundry_list = np.array(boundry_list)
            # boundry_list = boundry_list * 2 - 1
            boundry_list = find_cells(self.grid.grid)
            next_point = next_waypoint(self, boundry_list)

            if next_point != None:

                new_grid = Grid(size_area_world=self.size_area,
                                    resolution=8)
                new_grid.grid[next_point[0], next_point[1]] = 100

                new_zoomed_size = (int(new_grid.size_area_world[1] * 0.5),
                                    int(new_grid.size_area_world[0] * 0.5))
                zoomed_grid = cv2.resize(new_grid.grid, new_zoomed_size,
                                            interpolation=cv2.INTER_NEAREST)

                new_grid.display(zoomed_grid,
                                self.estimated_pose,
                                title="new waypoint")


            # self.grid.display(self.grid.grid,
            #                   self.estimated_pose,
            #                   title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")
            # pass

        return command
