from enum import Enum
import math
import sys
import os
import numpy as np
import cv2
from typing import Optional

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from solutions.my_example_mapping import OccupancyGrid
from spg_overlay.entities.drone_abstract import DroneAbstract

from solutions.pid_controller import PIDController
from solutions.shortest_path import shortest_path
from solutions.waypoint import find_cells, next_waypoint
from spg_overlay.utils.utils import circular_mean

class MyDroneEval(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1 
        # The default state where we don't know of any woneded people.
        # Drone explores its environment.
        
        GOING_TO_WOUNDED = 2
        # The drone knows where a wounded person is and does not have them in sight, but is going to pick them up.

        GRASPING_WOUNDED = 3
        # Have wounded person in sight and is going to pick them up.

        SEARCHING_RESCUE_CENTER = 4
        # The drone has a wounded person and does not know where the rescue center is.

        DROPPING_AT_RESCUE_CENTER = 5
        # The drone has a wounded person and knows where the rescue center is.
        
        APPRAOCHING_RESCUE_CENTER = 6
        # Drone has a wounded person and has rescue center in sight.

        RETURN_TO_AREA = 7
        # All wounded people have been rescued and the drone goes to return area.

    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)

        self.iteration: int = 0

        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        self.estimated_pose = Pose() # Estimated position and orientation of the drone (world coordinates)
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=30, #size of grid pixel in world meters
                                  lidar=self.lidar(),
                                  semantic=self.semantic())
        
        self.PID = PIDController(kp=[0, 0, 2], ki=[0, 0, 0], kd=[0, 0, 0])

        self.waypoint = None # Final destination in grid coordinates
        self.steps = np.array([]) # Next steps in grid coordinates

        self.return_aera = None # Position (grid coordinates) of the return area
        self.known_wounded = np.array([]) # List of positions (grid coordinates) of known wounded
        self.rescue_center_points = np.empty((0, 2)) # List of positions (grid coordinates) on the edge of a rescue center


    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        # TODO: Implement this method
        pass

    def control(self):
        """
        control loop
        """
        # TODO: remove wounded from known_wounded if they have been rescued and keep track of rescued people

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        # Increment the iteration counter
        self.iteration += 1

        if self.iteration % 5 == 0:
            self.show_debug()
            print(f"State: {self.state}, PID Integral: {self.PID.integral}")

        #############
        # Update what drone knows about its environement
        #############

        self.estimated_pose = self.get_pose()
        self.grid.update_grid(pose=self.estimated_pose)
        wounded, rescue = self.process_semantic() # Positions of wounded and rescue center points in direct sight

        # Remember return area
        if self.iteration == 1:
            # We know for sure we can use gps in the starting position
            self.return_aera = self.grid._conv_world_to_grid(*self.estimated_pose.position) 

        #############
        # Take decisions based on what drone knows
        #############

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        #TODO: Handle special zones

        if (self.state is self.Activity.SEARCHING_WOUNDED and 
            len(self.known_wounded) > 0):
            self.state = self.Activity.GOING_TO_WOUNDED

        elif (self.state is self.Activity.GOING_TO_WOUNDED and
              len(wounded) > 0):
            self.state = self.Activity.GRASPING_WOUNDED

        elif (self.state is self.Activity.GRASPING_WOUNDED and 
            self.grasped_entities()):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif (self.state is self.Activity.GRASPING_WOUNDED and 
            len(self.known_wounded) == 0):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.SEARCHING_RESCUE_CENTER and
              len(self.rescue_center_points) > 0):
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
                len(rescue) > 0):
            self.state = self.Activity.APPRAOCHING_RESCUE_CENTER

        elif (self.state is self.Activity.APPRAOCHING_RESCUE_CENTER and
              not self.grasped_entities()):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              len(self.rescue_center_points) == 0):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
        
        #############
        # COMMANDS FOR EACH STATE
        #############

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GOING_TO_WOUNDED:
            self.waypoint = self.known_wounded[0]
            command = self.go_to_waypoint()
            command["grasper"] = 1

        elif self.state is self.Activity.GRASPING_WOUNDED:
            wounded_pos = wounded[0][3:]
            target_yaw = math.atan2(wounded_pos[1] - self.estimated_pose.position[1], wounded_pos[0] - self.estimated_pose.position[0])
            command = self.PID.compute_control(drone_pos=self.estimated_pose,
                                               target_pos=wounded_pos,
                                               target_yaw=target_yaw)
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_explore()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = self.go_to_waypoint()
            command["grasper"] = 1

        elif self.state is self.Activity.APPRAOCHING_RESCUE_CENTER:
            best_angle = circular_mean(rescue[:, 2])
            command = self.PID.compute_control(drone_pos=self.estimated_pose,  
                                               target_pos=(0, 0),
                                               target_yaw=best_angle)
            command["forward"] = 1
            command["lateral"] = 0
            command["grasper"] = 1

        print(f"Command: {command}")
        return command

    def get_pose(self):
        """
        Calculate current position (world coordinates) and orientation of the drone 
        using gps or accelerometer and return a Pose object.
        """
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
        """
        Returns wounded and rescue center points (world coordinates) in direct sight, sorted by distance to drone.
        Also updates positions of known wounded people and rescue centers based on semantic sensor. 
        """
        detection_semantic = self.semantic_values()
        x_drone, y_drone = self.estimated_pose.position

        for data in detection_semantic:
            # Calculate world coorinates of the detected entity
            x_entity = x_drone + math.cos(data.angle + self.estimated_pose.orientation) * data.distance
            y_entity = y_drone + math.sin(data.angle + self.estimated_pose.orientation) * data.distance
            grid_point = list(self.grid._conv_world_to_grid(x_entity, y_entity))
            wounded = [] # wounded person positions (world coordinates)
            rescue = [] # rescue center points (world coordinates)

            # If drone sees a wounded person, update internal wounded list
            if (data.entity_type ==
                DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                not data.grasped):

                wounded.append((data.distance, data.angle, x_entity, y_entity))
                
                for pos, i in enumerate(self.known_wounded):
                    # Check if person already in list. If yes, update position
                    if math.dist(grid_point, pos) <= 2: #TODO: Define a threshold
                        self.known_wounded[i] = grid_point
                    else: # Otherwise, append to list
                        np.concatenate((self.known_wounded, [grid_point]))

            # If drone sees a rescue center, update internal rescue center position list
            elif (data.entity_type ==
                  DroneSemanticSensor.TypeEntity.RESCUE_CENTER and
                  not data.grasped):
                
                rescue.append((data.distance, data.angle, x_entity, y_entity))

                if grid_point not in self.rescue_center_points:
                    np.concatenate((self.rescue_center_points, [grid_point]))

        return np.sort(np.array(wounded)), np.sort(np.array(rescue))

    def control_explore(self, threshold=2):
        """
        Drone explores unknown regions of the map to find wounded people or rescue center.
        """
        # Check if waypoint is not set or if we have reached the waypoint
        reached_waypoint = math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(*self.waypoint)) < threshold*self.grid.resolution if self.waypoint != None else True
        if reached_waypoint:
            # Find new waypoint
            boundry_list = find_cells(self.grid.grid)
            self.waypoint = next_waypoint(self, boundry_list)
            
            # Reset steps so we can generate new ones in 'go_to_waypoint'
            self.steps = np.array([])

        return self.go_to_waypoint(threshold=threshold*2)
                
    def go_to_waypoint(self, threshold=2, look_back=False):
        """
        Generate sequence of steps and follow it to get to the drone's waypoint.\n
        :param treshold: ditance (grid units) to point to consider as reached\n
        :param look_back: if True, drone will look back while moving. Used while carying wounded person.\n
        """
        assert self.waypoint is not None, "Tried going to waypoint 'None'"
        if self.waypoint is None:
            return {"forward": 0, "lateral": 0, "rotation": 0}
        
        # if after finishing current steps, we still haven't reached the waypoint, generate new steps
        if len(self.steps) == 0:
            self.steps = shortest_path(self.grid._conv_world_to_grid(*self.estimated_pose.position), self.waypoint, self.grid)

        next_step = self.steps[0] # get next step
        next_step_world = self.grid._conv_grid_to_world(*next_step)
        # if we have reached the next step, remove it from the list
        if math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(*next_step_world)) < threshold*self.grid.resolution:
            next_step = self.steps.pop(0)

        # calculate the angle to the next step
        add = math.pi if look_back else 0
        target_yaw = math.atan2(next_step_world[1] - self.estimated_pose.position[1], next_step_world[0] - self.estimated_pose.position[0])
        print(next_step_world[1] - self.estimated_pose.position[1], next_step_world[0] - self.estimated_pose.position[0])
        print(target_yaw, self.estimated_pose.orientation)
        return self.PID.compute_control(drone_pos=self.estimated_pose,
                                        target_pos=next_step_world,
                                        target_yaw=add+target_yaw)
    
    def show_debug(self):
        """
        Show debug information on the screen
        """

        res = self.grid.resolution
        t_pose =  (res/2 * np.array(self.estimated_pose.position) + (res/4 - 0.5) * np.array([self.size_area[0], -self.size_area[1]]))

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