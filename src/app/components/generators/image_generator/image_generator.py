import os
import sys
import cv2 as cv
import numpy as np
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.constants.constants import *

class ImageGenerator(object):
    
    """This classs contains methods to create the image of the perfect model 
    of the 3D printed object
    
    Methods:
        generate_image (coords: List[List[object]]):
            Method to generate the image of the perfect model
    """
    
    @classmethod
    def generate_image(
            cls, 
            coords: List[List[object]],
            pixels_per_metric: float, 
            reference_object_width: float) -> np.ndarray:
        """Method to generate the image of the perfect model

        Parameters:
            coords (List[List[object]]): 
                Coordinates to create the image of the perfect model for the 
                3D printed object
            pixels_per_metric (float): 
                
        """
        
        blank_image = np.zeros(
            shape=hot_bed_shape, 
            dtype=np.float32)
        
        # For each layer
        for layer in coords:
            # For each perimeter that comprehends a layer
            for perimeter in layer[2]:
                for i in range(len(perimeter[1]) - 1):
                    cv.line(
                        blank_image, 
                        pt1=(perimeter[1][i][0], perimeter[1][i][1]),
                        pt2=(perimeter[1][i+1][0], perimeter[1][i+1][1]), 
                        color=perimeter_colors[perimeter[0]], 
                        thickness=round(perimeter[1][i+1][2]*1.5)
                    )
        
        cv.imshow("image", blank_image)
        k = cv.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv.destroyAllWindows()