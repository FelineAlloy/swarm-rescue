import os
import sys
# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)


from solutions.my_drone_random import MyDroneRandom
import math


class Node :
    def __init__(self, coordinates) :
        self.coordinates = coordinates # tuple of int
        self.neighbours = [] # list of neighbouring nodes of form Node

class Graph :
    def __init__(self) :
        self.room_map = {}
        
    def add_node(self, coordinates) :
        pass
        

    def shortest_path(self, start : Node, end : Node) :
        value_boxes = {i : float('inf') for i in self.room_map.keys()}
        value_boxes[start] = 0
        
           
        current = start
        path = [start]
        del value_boxes[current]
        
        while value_boxes :
            for i in current.neighbours :
                d = math.sqrt((start.coordinates[0] - end.coordinates[0])**2 + (start.coordinates[1] - end.coordinates[1])**2)
                if i in value_boxes and value_boxes[i] > value_boxes[current] + d :
                    value_boxes[i] += d
                        
            current = min(value_boxes, key=value_boxes.get)
            path.append(current)
            del value_boxes[current]
            
        return [i.coordinates for i in path]
    
class MyDroneEval(MyDroneRandom):
    pass
