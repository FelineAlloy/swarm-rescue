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
from spg_overlay.utils.timer import Timer
from spg_overlay.utils.constants import MAX_RANGE_SEMANTIC_SENSOR

from solutions.pid_controller import PIDController
from solutions.shortest_path import shortest_path
from solutions.waypoint import find_cells, next_waypoint
from spg_overlay.utils.utils import circular_mean, normalize_angle

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

        RETURN_TO_AREA = 8
        # Go to return area for various reasons (eg. communication)

        RETURN_TO_AREA_END = 8
        # All wounded people have been rescued and the drone goes to return area.

    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)

        self.iteration: int = 0

        # Flag for ehn the drone bno longer finds new cells to explore
        self.finished_exploring = False

        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # Maximum time without communication before going to return area
        self.max_time_communication = 30
        self.communication_timer = Timer(start_now=True)

        self.step_timeoput = 2 # Seconds to reach the next step on a path
        self.step_timer = Timer(start_now=True)

        self.estimated_pose = Pose() # Estimated position and orientation of the drone (world coordinates)
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=20, #size of grid pixel in world meters
                                  lidar=self.lidar(),
                                  semantic=self.semantic())
        
        self.PID = PIDController(kp=[1.6, 1.6, 7], ki=[0, 0, 0], kd=[11, 11, 0.6])

        self.waypoint = None # Final destination in grid coordinates
        self.steps = np.array([]) # Next steps in grid coordinates

        self.return_aera = None # Position (grid coordinates) of the return area
        self.known_wounded = np.empty((0, 2)) # List of positions (grid coordinates) of known wounded
        self.rescue_center_points = np.empty((0, 2)) # List of positions (grid coordinates) on the edge of a rescue center

        # Position of the wounded person the drone is going to grab when in line of sight
        self.wounded_pos = None
        # Sometimes the drone can lose line of sight with the person even though they are in range (idk why, maybe at the transition betweeen relying on known pos to going by visuals)
        # we use this variable to keep track of the last place we've seen the person. TODO: better way of doing this, maybe incorporating some locking onto a certain target
        # Same things goes for the rescue center
        self.rescue_pos = None

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier, self.grid, self.estimated_pose)
        
        return msg_data

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

        # if self.iteration % 5 == 0:
        #     self.show_debug()
        #     print(self.state)

        #############
        # Update what drone knows about its environement
        #############

        if self.odometer_values() is None: # drone is dead
            return command

        self.estimated_pose = self.get_pose()
        self.grid.update_grid(pose=self.estimated_pose)
        wounded, rescue, drones = self.process_semantic() # Positions of wounded and rescue center points in direct sight (world coordinates)

        # Remember return area
        if self.iteration == 1:
            # We know for sure we can use gps in the starting position
            self.return_aera = self.grid._conv_world_to_grid(*self.estimated_pose.position) 

        live_drones = self.process_communication_sensor()

        # Remove wounded from known_wounded if they are no longer there
        known_wounded_new = np.empty((0, 2))
        for pos in self.known_wounded:
            world_pos = self.grid._conv_grid_to_world(*pos)
            if math.dist(world_pos, self.estimated_pose.position) < MAX_RANGE_SEMANTIC_SENSOR:
                if len(wounded) < 0 or sum([math.dist(world_pos, w[2:]) < 2*self.grid.resolution for w in wounded]) != 0:
                    known_wounded_new = np.concatenate((known_wounded_new, [pos]))
        self.known_wounded = known_wounded_new
        if len(self.known_wounded) == 0:
            self.known_wounded.reshape((0, 2))

        # if self.communicator:
        #     for element in drones:
        #         drone_pos = element[2:]
        #         if not np.any(np.linalg.norm(live_drones - drone_pos, axis=1) < 2*self.grid.resolution):
        #             self.set_square_region(self.grid._conv_world_to_grid(*drone_pos), 2, 40)
                    
        #############
        # Take decisions based on what drone knows
        #############

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        # if self.finished_exploring: # temporary fix
        #     self.state = self.Activity.RETURN_TO_AREA

        if (self.state is self.Activity.SEARCHING_WOUNDED and
            len(self.known_wounded) > 0):
            self.state = self.Activity.GOING_TO_WOUNDED
    
        # if we haven't seen a drone in too long, then set next waypoint to the rescue zone
        elif (self.state is self.Activity.SEARCHING_WOUNDED and
            self.communication_timer.get_elapsed_time() > self.max_time_communication):
            self.state = self.Activity.RETURN_TO_AREA

        elif (self.state is self.Activity.RETURN_TO_AREA and 
              self.communication_timer.get_elapsed_time() < self.max_time_communication):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.GOING_TO_WOUNDED and
              len(wounded) > 0):
            self.state = self.Activity.GRASPING_WOUNDED

        elif (self.state is self.Activity.GOING_TO_WOUNDED and
              len(self.known_wounded) == 0):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.GRASPING_WOUNDED and 
            self.grasped_entities()):
            # When we grasp a person we remove them from the list of known wounded
            self.clear_grasped_entities()

            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif (self.state is self.Activity.GRASPING_WOUNDED and 
            len(self.known_wounded) == 0):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.SEARCHING_RESCUE_CENTER and
              len(self.rescue_center_points) > 0):
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
                len(rescue) > 2):
            self.state = self.Activity.APPRAOCHING_RESCUE_CENTER

        elif (self.state is self.Activity.APPRAOCHING_RESCUE_CENTER and
              not self.grasped_entities()):
            self.rescue_pos = None
            self.waypoint = None
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              len(self.rescue_center_points) == 0):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
        
        #############
        # COMMANDS FOR EACH STATE
        #############

        if self.state == self.Activity.RETURN_TO_AREA:
            self.waypoint = self.return_aera
            command = self.go_to_waypoint()
            command["grasper"] = 0

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GOING_TO_WOUNDED:
            if len(self.known_wounded) > 0:
                self.waypoint = self.known_wounded[0]
                command = self.go_to_waypoint()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            if len(wounded) > 0:
                self.wounded_pos = wounded[0][2:]
            if self.wounded_pos is not None:
                target_yaw = math.atan2(self.wounded_pos[1] - self.estimated_pose.position[1], self.wounded_pos[0] - self.estimated_pose.position[0])
                command = self.PID.compute_control(drone_pos=self.estimated_pose,
                                                target_pos=self.wounded_pos,
                                                target_yaw=target_yaw)
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_explore(look_back=True)
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            self.waypoint = self.rescue_center_points[0] 
            command = self.go_to_waypoint(look_back=True)
            command["grasper"] = 1

        elif self.state is self.Activity.APPRAOCHING_RESCUE_CENTER:
            if len(rescue) > 0:
                best_angle = circular_mean(rescue[:, 1])
                self.rescue_pos = rescue[np.argmin(np.abs(rescue[:, 1] - best_angle))][2:]
            elif self.rescue_pos is not None:
                best_angle = math.atan2(self.rescue_pos[1] - self.estimated_pose.position[1], self.rescue_pos[0] - self.estimated_pose.position[0])

            command = self.PID.compute_control(drone_pos=self.estimated_pose,
                                                target_pos=self.rescue_pos,
                                                target_yaw=math.atan2(self.rescue_pos[1] - self.estimated_pose.position[1], self.rescue_pos[0] - self.estimated_pose.position[0])) # TODO: i dont understant why this shouldn't get the drone's curent orintation added since best angle is relative to drone orientation (this is how the sensors give angles)
            # command["forward"] = 0
            # command["lateral"] = 0
            # if abs(best_angle) < math.pi/2:
            #     print(f"sending it: {best_angle}")
            #     command["forward"] = 1
            #     command["lateral"] = 0
            
            command["grasper"] = 1

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

        return Pose(position, orientation)

    def process_semantic(self):
        """
        Returns wounded and rescue center points (world coordinates) in direct sight, sorted by distance to drone.
        Also updates positions of known wounded people and rescue centers based on semantic sensor. 
        """
        detection_semantic = self.semantic_values()
        x_drone, y_drone = self.estimated_pose.position

        # Note: its kind of important for the implementetaion to have np arrays here
        wounded = np.empty((0, 4)) # wounded person positions (world coordinates)
        rescue = np.empty((0, 4)) # rescue center points (world coordinates)
        drones = np.empty((0, 4)) # other drones positions (world coordinates)

        for data in detection_semantic:
            # Calculate world coorinates of the detected entity
            x_entity = x_drone + math.cos(data.angle + self.estimated_pose.orientation) * data.distance
            y_entity = y_drone + math.sin(data.angle + self.estimated_pose.orientation) * data.distance
            grid_point = np.array(self.grid._conv_world_to_grid(x_entity, y_entity)).astype(int)

            # If drone sees a wounded person, update internal wounded list
            if (data.entity_type ==
                DroneSemanticSensor.TypeEntity.WOUNDED_PERSON):

                if 0 <= grid_point[0] < self.grid.x_max_grid and 0 <= grid_point[1] < self.grid.y_max_grid:
                    if not data.grasped:
                        wounded = np.append(wounded, [[data.distance, data.angle, x_entity, y_entity]], axis=0)
                        self.grid.grid[grid_point[0], grid_point[1]] = -0.6
                    else:
                        self.grid.grid[grid_point[0], grid_point[1]] = 0.6
                
                for i, pos in enumerate(self.known_wounded):
                    # Check if person already in list. If yes, update position
                    if math.dist(grid_point, pos) <= 2: #TODO: Define a threshold
                        # if the person is grasped, remove them from the list
                        if data.grasped:
                            self.known_wounded = np.delete(self.known_wounded, i, axis=0)
                        else:
                            self.known_wounded[i] = grid_point
                        break
                else:
                    # Otherwise, append to list
                    if len(self.known_wounded) == 0:
                        self.known_wounded.reshape((0, 2))
                    self.known_wounded = np.concatenate((self.known_wounded, [grid_point]))

            # If drone sees a rescue center, update internal rescue center position list
            elif (data.entity_type ==
                  DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                
                rescue = np.append(rescue, [[data.distance, data.angle, x_entity, y_entity]], axis=0)

                # we actually don't need to keep track of the points on the rescue center since those have a high probability of being walls
                # its actually better to just remember something close to the drone position (maybe weighted a little towards the rescue center)
                x_rescue = x_drone + math.cos(data.angle + self.estimated_pose.orientation) * data.distance * 0
                y_rescue = y_drone + math.sin(data.angle + self.estimated_pose.orientation) * data.distance * 0
                grid_point = np.array(self.grid._conv_world_to_grid(x_rescue, y_rescue)).astype(int)
                if grid_point not in self.rescue_center_points:
                    self.rescue_center_points = np.concatenate((self.rescue_center_points, [grid_point]))

            elif (data.entity_type ==
                  DroneSemanticSensor.TypeEntity.DRONE):
                drones = np.append(drones, [[data.distance, data.angle, x_entity, y_entity]], axis=0)

        return wounded[wounded[:, 0].argsort()], rescue[rescue[:, 0].argsort()], drones

    def control_explore(self, threshold=1, look_back=False):
        """
        Drone explores unknown regions of the map to find wounded people or rescue center.
        """
        # Check if waypoint is not set or if we have reached the waypoint
        if self.waypoint is None:
            reached_waypoint = True
        elif self.grid.grid[int(self.waypoint[0]), int(self.waypoint[1])] > -0.6:
            reached_waypoint = True
        else:
            reached_waypoint = math.dist(self.estimated_pose.position, self.grid._conv_grid_to_world(*self.waypoint)) < threshold*self.grid.resolution
        
        if reached_waypoint:
            # Find new waypoint
            boundry_list = find_cells(self.grid.grid)
            self.waypoint = next_waypoint(self, boundry_list) #TODO: Handle case where all map is explored
            
            # if self.waypoint is None:
            #     self.finished_exploring = True

            # Reset steps so we can generate new ones in 'go_to_waypoint'
            self.steps = np.empty(0)

        return self.go_to_waypoint(threshold=threshold*2, look_back=look_back)
                
    def go_to_waypoint(self, threshold=1, look_back=False):
        """
        Generate sequence of steps and follow it to get to the drone's waypoint.\n
        :param treshold: ditance (grid units) to point to consider as reached\n
        :param look_back: if True, drone will look back while moving. Used while carying wounded person.\n
        """
        # assert self.waypoint is not None, "Tried going to waypoint 'None'" # for debug purposes
        if self.waypoint is None:
            return {"forward": 0, "lateral": 0, "rotation": 0}
        
        # if after finishing current steps, we still haven't reached the waypoint, generate new steps
        if len(self.steps) == 0 or self.step_timer.get_elapsed_time() > self.step_timeoput:
            self.steps, self.waypoint = shortest_path(self.grid._conv_world_to_grid(*self.estimated_pose.position), self.waypoint, self.grid)
            # if the waypoint is not reachable, shortest_path will modify it


        next_step = self.steps[0] # get next step
        next_step_world = self.grid._conv_grid_to_world(*next_step)
        # if we have reached the next step, remove it from the list
        if math.dist(self.estimated_pose.position, next_step_world) < threshold*self.grid.resolution:
            next_step = self.steps.pop(0)
            self.step_timer.restart()

        # calculate the angle to the next step
        add = math.pi if look_back else 0
        target_yaw = math.atan2(next_step_world[1] - self.estimated_pose.position[1], next_step_world[0] - self.estimated_pose.position[0])
        return self.PID.compute_control(drone_pos=self.estimated_pose,
                                        target_pos=next_step_world,
                                        target_yaw=add+target_yaw)
    
    def clear_grasped_entities(self):
        """
        Clear the person that was just grasped from all lists.
        """
        self.wounded_pos = None

        drone_pos_grid = self.grid._conv_world_to_grid(*self.estimated_pose.position)
        # Compute Euclidean distances
        distances = np.linalg.norm(self.known_wounded - np.asarray(drone_pos_grid), axis=1)
        # Filter wounded that are more than 4 units away
        self.known_wounded = self.known_wounded[distances > 4] #TODO: Define a threshold

    def process_communication_sensor(self):
        
        live_drones = np.empty((0, 2))
        found_drone = False # If drone doesn't communicate enough, have it go to return area
        if self.communicator:
            received_messages = self.communicator.received_messages
            found_drone = len(received_messages) > 0

            for msg in received_messages:
                message = msg[1]
                self.grid.grid[self.grid.grid==0] = message[1].grid[self.grid.grid==0]
                live_drones = np.concatenate((live_drones, [message[2].position]))

        if found_drone :
            self.communication_timer.restart()

        return live_drones

    def set_square_region(self, position, radius, value):
        """
        Sets all values in a square around a given position in a NumPy grid.

        Parameters:
            grid (np.ndarray): The 2D NumPy array to modify.
            position (tuple): (row, col) coordinates of the center.
            radius (int): The half-size of the square region.
            value (int/float): The value to set in the square region.

        Returns:
            None (modifies grid in place).
        """
        row, col = position

        # Compute bounds while keeping them within grid limits
        row_start = max(row - radius, 0)
        row_end = min(row + radius + 1, self.grid.grid.shape[0])
        col_start = max(col - radius, 0)
        col_end = min(col + radius + 1, self.grid.grid.shape[1])

        # Set values in the square region
        self.grid.grid[row_start:row_end, col_start:col_end] = value

    def show_debug(self):
        """
        Show debug information on the screen
        """

        res = self.grid.resolution
        t_pose =  (res/2 * np.array(self.estimated_pose.position) + (res/4 - 0.5) * np.array([self.size_area[0], -self.size_area[1]]))

        self.grid.display(self.grid.zoomed_grid,
                            Pose(t_pose, self.estimated_pose.orientation),
                            title="zoomed occupancy grid")
        
        if self.waypoint is not None:

            # Make sure coordinates are casted to ints
            # failing to do so causes errors later on
            self.waypoint = list(map(int, self.waypoint))

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