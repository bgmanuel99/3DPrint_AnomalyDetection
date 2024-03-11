import os
import io
import sys
import turtle
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.constants.constants import *

class TurtleImageGenerator:
    
    """This classs contains methods to create the image of the perfect model 
    of the 3D printed object
    
    Methods:
        generate_image (coords: List[List[object]]):
            Method to generate the image of the perfect model
    """
    
    @classmethod
    def generate_image(cls, coords: List[List[object]]) -> io.TextIOWrapper:
        """Method to generate the image of the perfect model

        Parameters:
            coords (List[List[object]]): 
                Coordinates to create the image of the 3D printed object
        """
        
        nozzle = turtle.Turtle()
        nozzle.speed(nozzle_speed)
        
        # For each layer
        for layer in coords:
            # For each perimeter that comprehends a layer
            for perimeter in layer[2]:
                nozzle.pencolor(nozzle_colors[perimeter[0]])
                
                for i in range(len(perimeter[1])):
                    if perimeter[1][i][2] == 0.0:
                        nozzle.penup()
                        nozzle.goto(
                            perimeter[1][i][0],
                            perimeter[1][i][1])
                        nozzle.pendown()
                    else:
                        nozzle.pensize(perimeter[1][i][2])
                        nozzle.goto(
                            perimeter[1][i][0],
                            perimeter[1][i][1])