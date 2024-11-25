from enum import Enum
import math
import sys
import os
from pathlib import Path
from typing import Type
import numpy as np
import cv2

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)

from spg_overlay.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor

from solutions.my_example_mapping import MyDroneMapping

from solutions.movement import go_to
from solutions.waypoint import find_cells, next_waypoint
from solutions.shortest_path import shortest_path

class MyDrone(MyDroneMapping):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self, **kwargs):
        super().__init__(resolution=15, **kwargs)
        self.waypoint = None # final destination in grid coordinates
        self.steps = None # next steps in grid coordinates

        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # points on grid bellonging to a rescue center
        self.rescue = None #in grid coordinates

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

        if self.iteration == 1:
            self.return_area = self.grid._conv_world_to_grid(*self.measured_gps_position()) #in grid coordinates

        # evenetually we can make a better function which filters the noise from the sensors
        # for now we need to handle the case when we don't have gps
        # self.estimated_pose = self.get_pose()
        
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        
        # update map
        self.grid.update_grid(pose=self.estimated_pose)
        person, rescue = self.process_semantic()
        if rescue != (None, None):
            self.rescue = rescue

        

        # debugging views
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

        if self.state == self.Activity.SEARCHING_WOUNDED:
            if person != (None, None):
                self.state = self.Activity.GRASPING_WOUNDED 
                self.waypoint = person
                self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)
            else:    
                # serach wonded
                reached_waypoint = math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(self.waypoint[0], self.waypoint[1])) < 2*self.grid.resolution if self.waypoint != None else True #set better value here
                if reached_waypoint:
                    #find waypoint
                    boundry_list = find_cells(self.grid.grid)
                    self.waypoint = next_waypoint(self, boundry_list)
                
                    # get path to waypoint
                    self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)

                next_step = self.steps[0]
                if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1])) < 2*self.grid.resolution: #set better value here
                    next_step = self.steps.pop(0)

                command = go_to(self, self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        if self.state == self.Activity.GRASPING_WOUNDED:
            # rescue person
            if self.grasped_entities():
                if self.rescue:
                    self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
                    self.waypoint = self.rescue
                    self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)
                else:
                    self.state = self.Activity.SEARCHING_RESCUE_CENTER

            command["grasper"] = 1

            next_step = self.steps[0]
            if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1])) < 2*self.grid.resolution: #set better value here
                next_step = self.steps.pop(0)

            command = go_to(self, self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))
        
        elif self.state == self.Activity.SEARCHING_RESCUE_CENTER:
            if self.rescue != (None, None):
                self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
                self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
                self.waypoint = self.rescue
                self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)

            # serach rescue center
            reached_waypoint = math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(self.waypoint[0], self.waypoint[1])) < 2*self.grid.resolution #set better value here
            if self.waypoint == None or reached_waypoint:
                #find waypoint
                boundry_list = find_cells(self.grid.grid)
                self.waypoint = next_waypoint(self, boundry_list)
            
                # get path to waypoint
                self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)

            next_step = self.steps[0]
            if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1])) < 2*self.grid.resolution: #set better value here
                next_step = self.steps.pop(0)

            command = go_to(self, self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        elif self.state == self.Activity.DROPPING_AT_RESCUE_CENTER:
            if not self.grasped_entities():
                self.waypoint = self.rescue
                self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)

            # go to center
            next_step = self.steps[0]
            if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1])) < 2*self.grid.resolution: #set better value here
                next_step = self.steps.pop(0)

            command = go_to(self, self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command["grasper"] = 1

        return command

    def process_semantic(self):
        detection_semantic = self.semantic_values()

        if detection_semantic == None:
            return

        person = (None, None)
        rescue = (None, None)

        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                x, y = self.estimated_pose.position
                x_person = x + math.cos(data.angle) * data.distance
                y_person = y + math.sin(data.angle) * data.distance
                person = self.grid._conv_world_to_grid(x_person, y_person)

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                x, y = self.estimated_pose.position
                x_rescue = x + math.cos(data.angle) * data.distance
                y_rescue = y + math.sin(data.angle) * data.distance
                rescue = self.grid._conv_world_to_grid(x_rescue, y_rescue)

        return (person, rescue)

