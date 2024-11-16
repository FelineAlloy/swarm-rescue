import arcade
import math

class KeyboardController:

    def __init__(self):
        self.command = {"forward": 0.0,
                         "side": 0.0,
                         "rotation": 0.0,}
    def on_key_press(self, key):
        if key == arcade.key.UP:
            self.command["forward"] = 1.0
        elif key == arcade.key.DOWN:
            self.command["forward"] = -1.0

        if key == arcade.key.MOD_SHIFT:
            if key == arcade.key.LEFT:
                self.command["rotation"] = 1.0
            elif key == arcade.key.RIGHT:
                self.command["rotation"] = -1.0
        else:
            if key == arcade.key.LEFT:
                self.command["side"] = 1.0
            elif key == arcade.key.RIGHT:
                self.command["side"] = -1.0

    def on_key_release(self, key, modifiers):
        if key == arcade.key.UP:
            self.command["forward"] = 0
        elif key == arcade.key.DOWN:
            self.command["forward"] = 0

        if key == arcade.key.LEFT:
            self.command["side"] = 0
            self.command["rotation"] = 0
        elif key == arcade.key.RIGHT:
            self.command["side"] = 0
            self.command["rotation"] = 0

    def control(self):
        return self.command
# Two points and go from point A to point B

    def go_to(self, drone, point_A, point_B):
        # point_A and point_B are cartesion coordinates (tuples)
        _, b = drone.estimated_pose 
        epsilon_d = 0.5
        epsilon_a = 0.1

        target_angle = math.atan2((point_B[1]-point_A[1]),(point_B[0]-point_A[0]))
        if not (target_angle - epsilon_a < b < target_angle + epsilon_a):
            if not (-math.pi > target_angle - b > math.pi):
                self.command("rotation") = 1.0
            else:
                self.command("rotation") = -1.0
        else:
            self.command("rotation") = 0
            if not(point_B[0] - epsilon_d <= point_A[0] <= point_B[0] + epsilon_d ) and not(point_B[1] - epsilon_d <= point_A[1] <= point_B[1] + epsilon_d):
                self.command["forward"] = 1.0
            else:
                self.command["forward"] = 0
                return True