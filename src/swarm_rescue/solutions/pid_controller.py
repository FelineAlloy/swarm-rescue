import numpy as np

from spg_overlay.utils.utils import normalize_angle

class PIDController:
        def __init__(self, kp, ki, kd):
            self.kp = np.array(kp)  # Proportional gains (forward, lateral, rotation)
            self.ki = np.array(ki)  # Integral gains
            self.kd = np.array(kd)  # Derivative gains
            
            self.integral = np.zeros(3)
            self.prev_error = np.zeros(3)
            self.last_time = None

        def compute_control(self, drone_pos, target_pos, target_yaw, dt=1): #TODO: See if we actually need dt
            """
            Computes control commands for translation and rotation.\n
            :param drone_pos: Pose object of drone (world coordinates)\n
            :param target_pos: Tuple (x, y) for target position (world coordinates)\n
            :param target_yaw: Target yaw angle (radians)\n
            :param dt: Time step in seconds\n
            :return: Dictionary with 'forward', 'lateral', 'rotation' fields
            """
            # Compute position errors
            error_x = target_pos[0] - drone_pos.position[0]
            error_y = target_pos[1] - drone_pos.position[1]
            yaw = drone_pos.orientation
            
            # Rotate errors into the drone's local frame
            forward_error = error_x * np.cos(yaw) + error_y * np.sin(yaw)
            lateral_error = -error_x * np.sin(yaw) + error_y * np.cos(yaw)
            yaw_error = normalize_angle(target_yaw - yaw)
            
            errors = np.array([forward_error, lateral_error, yaw_error])
            
            # Compute integral term
            self.integral += errors * dt
            self.integral[2] = normalize_angle(self.integral[2])
            
            # Compute derivative term
            derivative = (errors - self.prev_error) / dt
            derivative[2] = normalize_angle(derivative[2])
            
            # PID output
            output = self.kp * errors + self.ki * self.integral + self.kd * derivative
            
            # Update previous error
            self.prev_error = errors
            
            # Clamp outputs to [-1, 1]
            output = np.clip(output, -1, 1)
            
            return {'forward': output[0], 'lateral': output[1], 'rotation': output[2]}