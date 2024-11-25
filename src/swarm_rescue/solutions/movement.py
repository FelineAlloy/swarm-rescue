import arcade
import math
import numpy as np

from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR

def go_to(drone, point_A, point_B):
    command = {"forward": 0.0,
                "lateral": 0.0,
                "rotation": 0.0,
                "grasper": 0}

    # point_A and point_B are cartesian coordinates (tuples)
    # b = drone.estimated_pose.orientation
    # epsilon_d = 0.5
    # epsilon_a = 0.2 * math.pi

    # target_angle = math.atan2((point_B[1]-point_A[1]),(point_B[0]-point_A[0]))
    # if not (target_angle - epsilon_a < b < target_angle + epsilon_a):
    #     if not (-math.pi > target_angle - b  > math.pi):
    #         command["rotation"] = 1
    #     else:
    #         command["rotation"] = 0
    # else:
    #     command["rotation"] = 0
    #     if not(point_B[0] - epsilon_d <= point_A[0] <= point_B[0] + epsilon_d ) and not(point_B[1] - epsilon_d <= point_A[1] <= point_B[1] + epsilon_d):
    #         command["forward"] = 1
    #     else:
    #         command["forward"] = -0.9
    
    deriv_diff_position = diff_position - self.prev_diff_position
    Kp = 1.6
    Kd = 11.0

    forward = (Kp * float(diff_position[0]) +
                Kd * float(deriv_diff_position[0]))

    forward = clamp(forward, -1.0, 1.0)

    print("counter", self.counter,
            ", diff_position", int(diff_position[0] * 10),
            "forward=", forward)

    rotation = 0.0
    command = {"forward": forward,
                "rotation": rotation}

    self.prev_diff_position = diff_position

    return command