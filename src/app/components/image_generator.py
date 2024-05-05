import cv2
import math
import copy
import numpy as np
from typing import List
from imutils import perspective

from app.utils.constants.constants import *
from app.common.common import CommonFunctionalities

class ImageGenerator(object):
    """This class contains methods to create the image of the perfect model 
    of the 3D printed object
    
    Methods:
        generate_images (
                image_shape: List[int], 
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
    def generate_images(
            cls, 
            image_shape: List[int], 
            middle_coords_3d_object: tuple[float], 
            top_left_coord_3d_object: tuple[float], 
            coords: List[List[object]],
            ppm_degree_offset: List[float], 
            reference_object_width: float) -> np.ndarray:
        """Method to generate the image of the perfect model

        Parameters:
            image_shape (List[int]): 
                Original image shape
            middle_coords_3d_object (tuple[float]): 
                Middle coordinates of the 3d printed object in the original 
                image
            top_left_coord_3d_object (tuple[float]): 
                Top left coordinates of the 3d printed object in the original
                image
            coords (List[List[object]]): 
                Real coordinates of the 3d printed object
            ppm_degree_offset (List[float]): 
                List of pixels per metric values variations representing degree
                offsets when taking the picture of the original image
            reference_object_width (float): Width of the reference object

        Returns:
            np.ndarray: Image with the perfect model of the 3d printed object
        """
        
        # List of perfect models
        perfect_models = []
        
        max_strand_width = 0
        
        for pixels_per_metric in ppm_degree_offset:
            perfect_model = np.zeros(shape=image_shape, dtype=np.uint8)
            
            for layer in coords:
                mean_coord = cls._get_mean_coord(layer)
                transformed_layer = cls._normalize_and_transform_coords(
                    layer, 
                    mean_coord, 
                    pixels_per_metric, 
                    reference_object_width, 
                    middle_coords_3d_object)
                
                actual_max_strand_width = cls._get_max_strand_width(
                    transformed_layer)
                
                if actual_max_strand_width > max_strand_width:
                    max_strand_width = actual_max_strand_width
                
                for perimeter in transformed_layer[2]:
                    for i in range(len(perimeter[1]) - 1):
                        cv2.line(
                            perfect_model, 
                            pt1=(perimeter[1][i][0], perimeter[1][i][1]),
                            pt2=(perimeter[1][i+1][0], perimeter[1][i+1][1]), 
                            color=(255, 255, 255), 
                            thickness=perimeter[1][i+1][2]
                        )
            
            perfect_models.append(perfect_model)
        
        # List of transformed perfect models
        translated_perfect_models = []
        
        for perfect_model in perfect_models:
            cnts = CommonFunctionalities.find_and_grab_contours(perfect_model)
            
            box = CommonFunctionalities.get_box_coordinates(cnts[0])
            (top_left, top_right, bottom_right, bottom_left) = perspective \
                .order_points(box)
                
            translated_perfect_model = CommonFunctionalities \
                .get_translated_object(
                    perfect_model, 
                    top_left, 
                    top_right, 
                    bottom_right, 
                    bottom_left, 
                    top_left_coord_3d_object, 
                    perfect_model.shape, 
                    math.ceil(max_strand_width/2))
            
            translated_perfect_models.append(translated_perfect_model)
        
        return translated_perfect_models
    
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
        
        transformed_layer = copy.deepcopy(layer)
        
        for perimeter in transformed_layer[2]:
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
        
        return transformed_layer
    
    @classmethod
    def _get_max_strand_width(cls, layer: List[object]) -> int:
        """Method to obtain the maximum strand width in pixels from the 
        extrusion data points

        Parameters:
            coords (List[List[object]]): 
                Transformed coordinates of the 3d printed object
                
        Returns:
            int: Maximum strand width
        """
        
        max_strand_width = 0
        
        for perimeter in layer[2]:
            for i in range(len(perimeter[1])):
                if perimeter[1][i][2] > max_strand_width:
                    max_strand_width = perimeter[1][i][2]
                else: 
                    continue
                    
        return max_strand_width