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

from solutions.movement import go_to
from solutions.waypoint import find_cells, next_waypoint

# temp, will have to sync with mapping
EVERY_N = 3
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 2.0
FREE_ZONE_VALUE = -4.0
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40

class MyDroneEval(MyDroneMapping):
    def __init__(self, **kwargs):
        super().__init__(resolution=15, **kwargs)
        self.waypoint = None

    def define_message_for_all(self):
        """
        To do later
        """
        pass

    def control(self):
        """
        Draft of control loop
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
        if self.iteration % 5 == 0:

            res = self.grid.resolution
            t_pose =  (res/2 * np.array(self.estimated_pose.position) + (res/4 - 0.5) * np.array([self.size_area[0], -self.size_area[1]]))

            # self.grid.display(self.grid.grid,
            #                   self.estimated_pose,
            #                   title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              Pose(t_pose, self.estimated_pose.orientation),
                              title="zoomed occupancy grid")
            
            if self.waypoint != None:

                new_grid = Grid(size_area_world=self.size_area, resolution=self.grid.resolution)
                new_grid.grid[self.waypoint[0], self.waypoint[1]] = 100
                new_grid.grid[*self.grid._conv_world_to_grid(0, 0)] = -100

                new_zoomed_size = (int(new_grid.size_area_world[1] * 0.5),
                                   int(new_grid.size_area_world[0] * 0.5))
                zoomed_grid = cv2.resize(new_grid.grid, new_zoomed_size,
                                         interpolation=cv2.INTER_NEAREST)

                new_grid.display(zoomed_grid,
                                Pose(t_pose, self.estimated_pose.orientation),
                                title="new waypoint")

        if self.waypoint == None:
            boundry_list = find_cells(self.grid.grid)
            self.waypoint = next_waypoint(self, boundry_list)
        elif math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(self.waypoint[0], self.waypoint[1])) < 2*self.grid.resolution: #set better value here
            boundry_list = find_cells(self.grid.grid)
            self.waypoint = next_waypoint(self, boundry_list)
        else:
            command = go_to(self, self.estimated_pose.position, self.grid._conv_grid_to_world(self.waypoint[0], self.waypoint[1]))

        # print(self.waypoint)

        return command
