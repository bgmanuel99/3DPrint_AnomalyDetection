import os
import sys
import cv2 as cv
import numpy as np
from typing import List
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.constants.constants import *

class ImageGenerator(object):
    
    """This class contains methods to create the image of the perfect model 
    of the 3D printed object
    
    Methods:
        generate_image (
                segmented_image: np.ndarray, 
                middle_coords_3d_object: tuple[float, float], 
                coords: List[List[object]],
                pixels_per_metric: float, 
                reference_object_width: float):
            Method to generate the image of the perfect model
        _get_mean_coord (layer: List[object]):
            Private method to calculate the mean value of all the coordinates 
            of one layer of the 3d printed object
        _metric_to_pixels (
                coord: float, 
                pixels_per_metric: float, 
                reference_object_width: float):
            Private method to transform a real world coordinate of the 3d 
            printed object to a pixel value based on the width of a reference 
            object in the original image and the number of pixels of that 
            width in the given image
        _normalize_and_transform_coords (
                layer: List[object], 
                mean_coord: float, 
                pixels_per_metric: float, 
                reference_object_width: float, 
                middle_coords_3d_object: tuple[float, float]):
            Private method to transform the real world coordinates of a layer 
            of the 3d printed object to pixel coordinates that can be used to 
            create the perfect model of the 3d printed object
    """
    
    @classmethod
    def generate_image(
            cls, 
            segmented_image: np.ndarray, 
            middle_coords_3d_object: tuple[float, float], 
            coords: List[List[object]],
            pixels_per_metric: float, 
            reference_object_width: float) -> np.ndarray:
        """Method to generate the image of the perfect model

        Parameters:
            segmented_image (np.ndarray): 
                Segmented image only with the 3d printed object in it
            middle_coords_3d_object (tuple[float, float]): 
                Middle coordinates of the 3d printed object in the original 
                image
            coords (List[List[object]]): 
                Real coordinates of the 3d printed object
            pixels_per_metric (float): 
                Pixels per metric value given by the width of the reference 
                object
            reference_object_width (float): Width of the reference object

        Returns:
            np.ndarray: Image with the perfect model of the 3d printed object
        """
        
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
    def _get_mean_coord(cls, layer: List[object]) -> float:
        """Method to calculate the mean value of all the coordinates of one 
        layer of the 3d printed object

        Parameters:
            layer (List[object]): 
                A list with all the coordinates of one layer of the 3d printed 
                object

        Returns:
            float: Mean value from the coordinates of the layer
        """
        
        max_coord = max(
            [number for sublist in layer[2] 
             for subsublist in sublist[1] 
             for number in subsublist[0:2]])
        min_coord = min(
            [number for sublist in layer[2] 
             for subsublist in sublist[1] 
             for number in subsublist[0:2]])

        return (max_coord + min_coord) / 2
    
    @classmethod
    def _metric_to_pixels(
            cls, 
            coord: float, 
            pixels_per_metric: float, 
            reference_object_width: float) -> float:
        """Method to transform a real world coordinate of the 3d printed 
        object to a pixel value based on the width of a reference object in 
        the original image and the number of pixels of that width in the given
        image

        Parameters:
            coord (float): Coordinate in a layer of the 3d printed object
            pixels_per_metric (float): 
                Pixels per metric value given by the width of the reference 
                object
            reference_object_width (float): Width of the reference object

        Returns:
            float: 
                Number of pixels for the a real world coordinate of the 3d 
                printed object
        """
        
        return coord * pixels_per_metric / reference_object_width
    
    @classmethod
    def _normalize_and_transform_coords(
            cls, 
            layer: List[object], 
            mean_coord: float, 
            pixels_per_metric: float, 
            reference_object_width: float, 
            middle_coords_3d_object: tuple[float, float]) -> List[object]:
        """Method to transform the real world coordinates of a layer of the 3d 
        printed object to pixel coordinates that can be used to create the 
        perfect model of the 3d printed object

        Parameters:
            layer (List[object]): 
                A list with all the coordinates of one layer of the 3d printed 
                object
            mean_coord (float): Mean value from the coordinates of the layer
            pixels_per_metric (float): 
                Pixels per metric value given by the width of the reference 
                object
            reference_object_width (float): Width of the reference object
            middle_coords_3d_object (tuple[float, float]): 
                Middle coordinates of the 3d printed object in the original 
                image

        Returns:
            List[object]: 
                A transformed layer with pixel value coordinates to create 
                the perfect model of the 3d printed object
        """
        
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