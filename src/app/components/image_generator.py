import os
import sys
import cv2 as cv
import numpy as np
from typing import List
import matplotlib.pyplot as plt

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
            segmented_image: np.ndarray, 
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
        
        max_coord, min_coord = 0.0, 0.0
        
        # For each layer
        for layer in coords:
            # For each perimeter that comprehends a layer
            max_coord = max([number for sublist in layer[2] for subsublist in sublist[1] for number in subsublist[0:2]])
            min_coord = min([number for sublist in layer[2] for subsublist in sublist[1] for number in subsublist[0:2]])

        mean_coord = (max_coord + min_coord) / 2
        
        # For each layer
        for layer in coords:
            # For each perimeter that comprehends a layer
            for perimeter in layer[2]:
                for i in range(len(perimeter[1])):
                    perimeter[1][i][0] = perimeter[1][i][0] - mean_coord
                    perimeter[1][i][1] = perimeter[1][i][1] - mean_coord
        
        middle_coord_image = cls._get_middle_coords(segmented_image)
        
        blank_image = np.zeros(
            shape=segmented_image.shape, 
            dtype=np.uint8)
        
        # For each layer
        for layer in coords:
            # For each perimeter that comprehends a layer
            for perimeter in layer[2]:
                for i in range(len(perimeter[1]) - 1):
                    cv.line(
                        blank_image, 
                        pt1=(
                            int(middle_coord_image[1] + cls._metric_to_pixels(
                                perimeter[1][i][0], 
                                pixels_per_metric, 
                                reference_object_width)), 
                            int(middle_coord_image[0] + cls._metric_to_pixels(
                                perimeter[1][i][1], 
                                pixels_per_metric, 
                                reference_object_width))),
                        pt2=(
                            int(middle_coord_image[1] + cls._metric_to_pixels(
                                perimeter[1][i+1][0], 
                                pixels_per_metric, 
                                reference_object_width)), 
                            int(middle_coord_image[0] + cls._metric_to_pixels(
                                perimeter[1][i+1][1], 
                                pixels_per_metric, 
                                reference_object_width))), 
                        color=perimeter_colors[perimeter[0]], 
                        thickness=1
                    )
        
        plt.imshow(blank_image, cmap="gray")
        plt.show()
        
        return blank_image
            
    @classmethod
    def _metric_to_pixels(
        cls, 
        coord: float, 
        pixels_per_metric: float, 
        reference_object_width: float) -> int:
        return int(coord * pixels_per_metric / reference_object_width)
    
    @classmethod
    def _get_middle_coords(cls, image):
        return (image.shape[0]/2, image.shape[1]/2)