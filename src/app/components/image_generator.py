import os
import sys
import cv2
import math
import imutils
import numpy as np
from typing import List
from imutils import perspective

sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.constants.constants import *
from app.common.common import print_image

class ImageGenerator(object):
    
    """This class contains methods to create the image of the perfect model 
    of the 3D printed object
    
    Methods:
        generate_image (
                image: np.ndarray, 
                middle_coords_3d_object: tuple[float], 
                top_left_coord_3d_object: tuple[float], 
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
                middle_coords_3d_object: tuple[float]):
            Private method to transform the real world coordinates of a layer 
            of the 3d printed object to pixel coordinates that can be used to 
            create the perfect model of the 3d printed object
    """
    
    @classmethod
    def generate_image(
            cls, 
            image: np.ndarray, 
            middle_coords_3d_object: tuple[float], 
            top_left_coord_3d_object: tuple[float], 
            coords: List[List[object]],
            pixels_per_metric: float, 
            reference_object_width: float) -> np.ndarray:
        """Method to generate the image of the perfect model

        Parameters:
            image (np.ndarray): 
                Original image
            middle_coords_3d_object (tuple[float]): 
                Middle coordinates of the 3d printed object in the original 
                image
            top_left_coord_3d_object (tuple[float]): 
                Top left coordinates of the 3d printed object in the original
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
        
        perfect_model = np.zeros(
            shape=image.shape[0:2], 
            dtype=np.uint8)
        
        max_strand_width = 0
        
        for layer in coords:
            mean_coord = cls._get_mean_coord(layer)
            layer = cls._normalize_and_transform_coords(
                layer, 
                mean_coord, 
                pixels_per_metric, 
                reference_object_width, 
                middle_coords_3d_object)
            
            print(layer)
            
            for perimeter in layer[2]:
                for i in range(len(perimeter[1]) - 1):
                    cv2.line(
                        perfect_model, 
                        pt1=(perimeter[1][i][0], perimeter[1][i][1]),
                        pt2=(perimeter[1][i+1][0], perimeter[1][i+1][1]), 
                        color=(255, 255, 255), 
                        thickness=perimeter[1][i+1][2]
                    )
                    max_strand_width = perimeter[1][i+1][2] if perimeter[1][i+1][2] > max_strand_width else max_strand_width
        
        print_image("perfect model", perfect_model, 600)
        
        cnts = cv2.findContours(
            perfect_model, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        box = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (tl, tr, br, bl) = box
        
        tl = list(map(int, tl))
        tr = list(map(int, tr))
        br = list(map(int, br))
        bl = list(map(int, bl))
        print(tl, tr, br, bl)
        
        perfect_model_external_contour = perfect_model[
            max(tl[1], tr[1]):max(br[1], bl[1]), 
            max(tl[0], bl[0]):max(tr[0], br[0])]
        print(perfect_model_external_contour.shape)
        
        print_image("External contour", perfect_model_external_contour, 600)
        
        transformed_perfect_model = np.zeros(
            perfect_model.shape, 
            dtype=np.uint8)
        
        x_offset = round(
            top_left_coord_3d_object[0] 
            + math.ceil(max_strand_width/2))
        y_offset = round(
            top_left_coord_3d_object[1] 
            + math.ceil(max_strand_width/2))
        
        x_end = x_offset + perfect_model_external_contour.shape[1]
        y_end = y_offset + perfect_model_external_contour.shape[0]
        
        transformed_perfect_model[
            y_offset:y_end, 
            x_offset:x_end] = perfect_model_external_contour
        
        print_image("Transformed model", transformed_perfect_model, 600)
        
        return transformed_perfect_model
    
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
            middle_coords_3d_object: tuple[float]) -> List[object]:
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
            middle_coords_3d_object (tuple[float]): 
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
                                           - perimeter[1][i][0])
                
                perimeter[1][i][1] = perimeter[1][i][1] - mean_coord
                perimeter[1][i][1] = cls._metric_to_pixels(
                    perimeter[1][i][1], 
                    pixels_per_metric, 
                    reference_object_width)
                perimeter[1][i][1] = round(middle_coords_3d_object[1] 
                                           - perimeter[1][i][1])
                
                perimeter[1][i][2] = round(cls._metric_to_pixels(
                    perimeter[1][i][2], 
                    pixels_per_metric, 
                    reference_object_width))
        
        return layer