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
    b = drone.estimated_pose.orientation
    epsilon_d = 0.5
    epsilon_a = 0.2 * math.pi

    target_angle = math.atan2((point_B[1]-point_A[1]),(point_B[0]-point_A[0]))
    if not (target_angle - epsilon_a < b < target_angle + epsilon_a):
        if not (-math.pi > target_angle - b  > math.pi):
            command["rotation"] = 1
        else:
            command["rotation"] = 0
    else:
        command["rotation"] = 0
        if not(point_B[0] - epsilon_d <= point_A[0] <= point_B[0] + epsilon_d ) and not(point_B[1] - epsilon_d <= point_A[1] <= point_B[1] + epsilon_d):
            command["forward"] = 1
        else:
            command["forward"] = -0.9
    
    # add a repelling force coming from obstacles - work in progerss
    # lidar_dist = drone.lidar_values()[::3].copy()
    # lidar_angles = drone.lidar_rays_angles()[::3].copy()

    # # Compute cos and sin of the absolute angle of the lidar
    # cos_rays = np.cos(lidar_angles + drone.estimated_pose.orientation)
    # sin_rays = np.sin(lidar_angles + drone.estimated_pose.orientation)

    # max_range = MAX_RANGE_LIDAR_SENSOR * 0.5

    # select = lidar_dist < max_range
    # if sum(select) > 0:
    #     magnitudes = 10**2 * lidar_dist[select] ** -2
    #     command["forward"] += sum(magnitudes * cos_rays[select]) / len(magnitudes) * 3
    #     command["forward"] /= 4
    #     command["lateral"] += sum(magnitudes * sin_rays[select]) / len(magnitudes) * 3
    #     command["lateral"] /= 4

    #     print(lidar_dist[select])

    return command