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
            middle_coords_3d_object, 
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
        
        middle_coord_image = cls._get_middle_coords(segmented_image)
        
        blank_image = np.zeros(
            shape=segmented_image.shape[0:2], 
            dtype=np.uint8)
        
        for layer in coords:
            mean_coord = cls._get_mean_coord(layer)
            layer = cls._normalize_and_transform_coords(
                layer, 
                mean_coord, 
                pixels_per_metric, 
                reference_object_width, 
                middle_coords_3d_object)
            print("layer")
            print(layer)
            
            for perimeter in layer[2]:
                for i in range(len(perimeter[1]) - 1):
                    cv.line(
                        blank_image, 
                        pt1=(perimeter[1][i][0], perimeter[1][i][1]),
                        pt2=(perimeter[1][i+1][0], perimeter[1][i+1][1]), 
                        color=(255, 255, 255), 
                        thickness=perimeter[1][i+1][2]
                    )
        
        plt.imshow(blank_image, cmap="gray")
        plt.show()
        
        return blank_image
    
    @classmethod
    def _get_middle_coords(cls, image):
        return (image.shape[0]/2, image.shape[1]/2)
    
    @classmethod
    def _get_mean_coord(cls, layer):
        max_coord = max([number for sublist in layer[2] for subsublist in sublist[1] for number in subsublist[0:2]])
        min_coord = min([number for sublist in layer[2] for subsublist in sublist[1] for number in subsublist[0:2]])

        return (max_coord + min_coord) / 2
    
    @classmethod
    def _metric_to_pixels(
        cls, 
        coord: float, 
        pixels_per_metric: float, 
        reference_object_width: float) -> int:
        return coord * pixels_per_metric / reference_object_width
    
    @classmethod
    def _normalize_and_transform_coords(
            cls, 
            layer, 
            mean_coord, 
            pixels_per_metric, 
            reference_object_width, 
            middle_coords_3d_object):
        for perimeter in layer[2]:
            for i in range(len(perimeter[1])):
                perimeter[1][i][0] = perimeter[1][i][0] - mean_coord
                perimeter[1][i][0] = cls._metric_to_pixels(
                    perimeter[1][i][0], 
                    pixels_per_metric, 
                    reference_object_width)
                perimeter[1][i][0] = round(middle_coords_3d_object[0] 
                                         + perimeter[1][i][0])
                
                perimeter[1][i][1] = perimeter[1][i][1] - mean_coord
                perimeter[1][i][1] = cls._metric_to_pixels(
                    perimeter[1][i][1], 
                    pixels_per_metric, 
                    reference_object_width)
                perimeter[1][i][1] = round(middle_coords_3d_object[1] 
                                         + perimeter[1][i][1])
                
                perimeter[1][i][2] = round(cls._metric_to_pixels(
                    perimeter[1][i][2], 
                    pixels_per_metric, 
                    reference_object_width))
        
        return layer