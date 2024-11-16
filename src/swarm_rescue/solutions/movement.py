import arcade
import math

def go_to(drone, point_A, point_B):
    command = {"forward": 0.0,
                "lateral": 0.0,
                "rotation": 0.0,
                "grasper": 0}
    
    # point_A and point_B are cartesion coordinates (tuples)
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
    
    return command