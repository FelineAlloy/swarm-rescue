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
from solutions.my_drone_lidar_communication import MyDroneLidarCommunication

from solutions.waypoint import find_cells, next_waypoint
from solutions.shortest_path import shortest_path
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.utils import normalize_angle

# todo: handle errors better, make the code more readable

class MyDrone(MyDroneMapping):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4
        RETURN_TO_AREA = 5 

    def __init__(self, **kwargs):
        super().__init__(resolution=30, **kwargs)
        self.last_communication = 0

        self.waypoint = None # final destination in grid coordinates
        self.steps = None # next steps in grid coordinates

        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # points on grid bellonging to a rescue center
        self.rescue = None #in grid coordinates

        # displacement used to calculate PID
        self.prev_diff_position = 0
        self.prev_diff_angle = 0

        # last command sent, used to estimate position when gps is not available
        self.prev_command = None

        # grasper toggle
        self.grasper = 0

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier, self.grid,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        
        #return msg_data
    

    def control(self):
        """
        Draft of control loop
        """

        max_time_communication = 10

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        

        if self.iteration == 1:
            self.return_area = self.grid._conv_world_to_grid(*self.measured_gps_position()) #in grid coordinates

        # evenetually we can make a better function which filters the noise from the sensors
        # for now we need to handle the case when we don't have gps
        self.estimated_pose = self.get_pose()
        
        # update map
        self.grid.update_grid(pose=self.estimated_pose)
        person, rescue, rescue_actual = self.process_semantic()
        if rescue != (None, None):
            self.rescue = rescue


        # debugging views
        # if self.iteration % 5 == 0:

        #     res = self.grid.resolution
        #     t_pose =  (res/2 * np.array(self.estimated_pose.position) + (res/4 - 0.5) * np.array([self.size_area[0], -self.size_area[1]]))

        #     # self.grid.display(self.grid.grid,
        #     #                   self.estimated_pose,
        #     #                   title="occupancy grid")
        #     self.grid.display(self.grid.zoomed_grid,
        #                       Pose(t_pose, self.estimated_pose.orientation),
        #                       title="zoomed occupancy grid")
            
        #     if self.waypoint != None:

        #         new_grid = Grid(size_area_world=self.size_area, resolution=self.grid.resolution)
        #         new_grid.grid[self.waypoint[0], self.waypoint[1]] = 100
        #         new_grid.grid[*self.grid._conv_world_to_grid(0, 0)] = -100

        #         new_zoomed_size = (int(new_grid.size_area_world[1] * 0.5),
        #                            int(new_grid.size_area_world[0] * 0.5))
        #         zoomed_grid = cv2.resize(new_grid.grid, new_zoomed_size,
        #                                  interpolation=cv2.INTER_NEAREST)

        #         new_grid.display(zoomed_grid,
        #                         Pose(t_pose, self.estimated_pose.orientation),
        #                         title="new waypoint")

        found_drone, _ = MyDroneLidarCommunication.process_communication_sensor()

        # update last_communication to current iteration if we have seen a drone in the current iteration
        if found_drone :
            self.last_communication = self.iteration

        # if we haven't seen a drone in too long, then set next waypoint to the rescue zone
        if self.iteration - self.last_communication > max_time_communication :
            self.state = self.Activity.RETURN_TO_AREA

        if self.state == self.Activity.SEARCHING_WOUNDED:
            if person != (None, None):
                self.state = self.Activity.GRASPING_WOUNDED 
                self.waypoint = person
                self.steps = []
                # self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)
            else:    
                # search wounded
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

                command = self.go_to(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        if self.state == self.Activity.GRASPING_WOUNDED:
            # rescue person
            if self.grasped_entities():
                if self.rescue:
                    self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
                    self.waypoint = self.rescue
                    self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)
                else:
                    self.state = self.Activity.SEARCHING_RESCUE_CENTER

            reached_waypoint = math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(self.waypoint[0], self.waypoint[1])) < 2*self.grid.resolution if self.waypoint != None else True #set better value here

            command.update(self.translate_to(self.grid._conv_grid_to_world(*self.waypoint)))
            if person != (None, None):
                command.update(self.translate_to(self.grid._conv_grid_to_world(*person)))
                command.update(self.point_to_angle(self.grid._conv_grid_to_world(*person)))
            
        elif self.state == self.Activity.SEARCHING_RESCUE_CENTER:
            if self.rescue != (None, None):
                self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
                self.waypoint = self.rescue
                self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)

            # search rescue center
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

            command = self.go_to(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        elif self.state == self.Activity.DROPPING_AT_RESCUE_CENTER:
            if not self.grasped_entities():
                self.state = self.Activity.RETURN_TO_AREA
                if self.return_area:
                    self.waypoint = self.return_area
                    self.steps = shortest_path(self.estimated_pose.position, self.waypoint, self.grid)
            
            # if rescue center in sight, head straight to it
            if rescue_actual != (None, None):
                # command = self.go_to(self.estimated_pose.position, self.grid._conv_grid_to_world(*rescue_actual))
                rescue_gps = self.grid._conv_grid_to_world(*rescue_actual)
                target_angle = math.atan2((rescue_gps[1]-self.estimated_pose.position[1]),(rescue_gps[0]-self.estimated_pose.position[0]))
                diff_angle = normalize_angle(target_angle - self.estimated_pose.orientation)
                forward = math.cos(diff_angle)
                lateral = math.sin(diff_angle)
                # command.update(self.point_to_angle(rescue_gps))
                command.update({
                    "forward": forward,
                    "lateral": lateral
                })

                print(self.prev_command)

            else:
                # go to last known position of rescue center
                if len(self.steps) > 0:
                    next_step = self.steps[0]
                    if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1])) < 1*self.grid.resolution: #set better value here
                        next_step = self.steps.pop(0)

                    command = self.go_to(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        elif self.state == self.Activity.RETURN_TO_AREA:
            reached_waypoint = math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(self.waypoint[0], self.waypoint[1])) < 2*self.grid.resolution if self.waypoint != None else True #set better value here
            if not reached_waypoint and len(self.steps) > 0:
                next_step = self.steps[0]
                if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1])) < 0.75*self.grid.resolution: #set better value here
                    next_step = self.steps.pop(0)

                command = self.go_to(self.estimated_pose.position, self.grid._conv_grid_to_world(next_step[0], next_step[1]))

        if self.state is self.Activity.SEARCHING_WOUNDED:
            self.grasper = 0
        else:
            self.grasper = 1

        command["grasper"] = self.grasper

        self.prev_command = command
        return command
    
    def process_communication(self):
        if self.communicator:
            received_messages = self.communicator.received_messages
            grid = received_messages[1]
            self.grid.grid = (self.grid.grid + grid.grid) / 2

    def get_pose(self):
        position = np.zeros(2)
        orientation = 0 

        if not self.compass_is_disabled():
            orientation = self.measured_compass_angle()
        elif self.estimated_pose:
            orientation = self.estimated_pose.orientation + self.odometer_values()[2]

        if not self.gps_is_disabled():
            position = np.array(self.measured_gps_position())
        elif self.estimated_pose:
            dist = self.odometer_values()[0]
            alpha = self.odometer_values()[1]

            dx = dist * math.cos(alpha + self.estimated_pose.orientation)
            dy = dist * math.sin(alpha + self.estimated_pose.orientation)
            position = self.estimated_pose.position + np.asarray([dx, dy])
        
        # print(self.true_position() - position, self.true_angle() - orientation)

        return Pose(position, orientation)

    def process_semantic(self):
        detection_semantic = self.semantic_values()

        if detection_semantic == None:
            return

        person = (None, None)
        rescue = (None, None)
        rescue_actual = (None, None)

        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                x, y = self.estimated_pose.position
                x_person = x + math.cos(data.angle + self.estimated_pose.orientation) * data.distance
                y_person = y + math.sin(data.angle + self.estimated_pose.orientation) * data.distance
                person = self.grid._conv_world_to_grid(x_person, y_person)

                self.grid.add_points(x_person, y_person, -40)

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                x, y = self.estimated_pose.position
                x_rescue = x + math.cos(data.angle + self.estimated_pose.orientation) * data.distance * 0.8
                y_rescue = y + math.sin(data.angle + self.estimated_pose.orientation) * data.distance * 0.8
                x_rescue_act = x + math.cos(data.angle + self.estimated_pose.orientation) * data.distance
                y_rescue_act = y + math.sin(data.angle + self.estimated_pose.orientation) * data.distance
                rescue = self.grid._conv_world_to_grid(x_rescue, y_rescue)
                rescue_actual = self.grid._conv_world_to_grid(x_rescue_act, y_rescue_act)

        return (person, rescue, rescue_actual)

    def point_to_angle(self, point_B):
        target_angle = math.atan2((point_B[1]-self.estimated_pose.position[1]),(point_B[0]-self.estimated_pose.position[0]))

        diff_angle = normalize_angle(target_angle - self.estimated_pose.orientation)

        deriv_diff_angle = normalize_angle(diff_angle - self.prev_diff_angle)
        Kp = 9.0
        Kd = 0.6
        rotation = Kp * diff_angle + Kd * deriv_diff_angle

        rotation = clamp(rotation, -1.0, 1.0)
        # print("counter", self.iteration, "angle", self.true_angle(),
        #       "diff_angle", diff_angle, "deriv_diff_angle", deriv_diff_angle,
        #       "sign(diff_angle)", math.copysign(1.0, diff_angle))

        command = {"rotation": rotation}

        self.prev_diff_angle = diff_angle

        return command
    
    def translate_to(self, point_B):
        
        target_angle = math.atan2((point_B[1]-self.estimated_pose.position[1]),(point_B[0]-self.estimated_pose.position[0]))

        angle_difference = self.estimated_pose.orientation - target_angle
        
        diff_position = (math.dist(self.estimated_pose.position, np.asarray(point_B)))

        deriv_diff_position = diff_position - self.prev_diff_position
        Kp = 1.6
        Kd = 11.0

        force = (Kp * float(diff_position) +
                    Kd * float(deriv_diff_position))

        force = clamp(force, -1.0, 1.0)

        # print("counter", self.iteration,
        #         ", diff_position", int(diff_position * 10),
        #         " forward ", force*math.cos(angle_difference),
        #         " lateral ", -force*math.sin(angle_difference),)

        command = {"forward": force*math.cos(angle_difference),
                   "lateral" : -force*math.sin(angle_difference)}

        self.prev_diff_position = diff_position

        return command

    def go_to(self, point_A, point_B):
        command = {"forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0}

        command.update(self.translate_to(point_B))        
        command.update(self.point_to_angle(point_B))

        return command
