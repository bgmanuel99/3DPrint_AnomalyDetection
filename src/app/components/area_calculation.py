import cv2
import math
import numpy as np
from typing import List
from imutils import contours

from app.common.common import (
    CommonPrints, 
    CommonFunctionalities, 
    CommonMorphologyOperations)

class AreaCalculation(object):
    """Class containing methods to calculate internal areas of the 3d printed
    object

    Methods:
        calculate_areas (
                masked_3d_object: np.ndarray, 
                reference_object_width: float, 
                reference_object_pixels_area: float):
            Method to retrieve an image with the internal enumerated contours 
            of the 3d printed object and a list of list with the corresponding
            areas calculated in square millimeters
        _find_and_sort_contours (
                segmented_image: np.ndarray):
            Private method to find and sort the internal contours of the 3d 
            printed object
        _draw_and_enumerate_contours (
                masked_3d_object_shape, 
                cnts: tuple[np.ndarray]):
            Private method to draw and enumerate the internal contours of the 
            3d printed object
        _retrieve_contours_pixels_areas (
                cnts: tuple[np.ndarray]):
            Private method to calculate the internal 3d printed object contour 
            areas
        _convert_areas_from_pixels_to_millimeters_squared (
                reference_object_width: float, 
                reference_object_pixels_area: float, 
                infill_pixels_areas: List[List[object]]):
            Private method to convert the areas in pixels to millimeters 
            squared
    """

    @classmethod
    def calculate_areas(
            cls, 
            masked_3d_object: np.ndarray, 
            reference_object_width: float, 
            reference_object_pixels_area: float) -> tuple[
                np.ndarray, List[List[object]]]:
        """Method to retrieve an image with the internal enumerated contours 
        of the 3d printed object and a list of list with the corresponding
        areas calculated in square millimeters

        Parameters:
            masked_3d_object (np.ndarray): 
                Transformed and masked image of the 3d printed object
            reference_object_width (float): 
                Known real width of the reference object
            reference_object_pixels_area (float): 
                Reference object area in pixels

        Returns:
            tuple[np.ndarray, List[List[object]]]: 
                - Image with the interenal enumerated contours
                - List of lists with internal 3d printed object contours areas 
                in millimeters squared
        """
        
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        opening = CommonMorphologyOperations.morphologyEx_opening(
            segmented_3d_object, (5, 5))
        
        CommonPrints.print_image("opening", opening, 600, True)

        cnts = cls._find_and_sort_contours(opening)

        infill_contours_image = cls._draw_and_enumerate_contours(
            masked_3d_object.shape, cnts)
        
        infill_pixels_areas = cls._retrieve_contours_pixels_areas(cnts)

        CommonPrints.print_image(
            "infill contours image", infill_contours_image, 600, True)
        
        infill_areas = cls._convert_areas_from_pixels_to_millimeters_squared(
            reference_object_width, 
            reference_object_pixels_area, 
            infill_pixels_areas)
            
        return infill_contours_image, infill_areas
    
    @classmethod
    def _find_and_sort_contours(
            cls, 
            segmented_image: np.ndarray) -> tuple[np.ndarray]:
        """Method to find and sort the internal contours of the 3d printed 
        object

        Parameters:
            segmented_image (np.ndarray): 
                Segmented image of the 3d printed object

        Returns:
            tuple[np.ndarray]: 
                Internal sorted contours of the 3d printed object
        """
        
        cnts, _ = cv2.findContours(
            segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from max to min by area
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]

        # Sort contours from left to right and top to bottom
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")
        cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")
        
        return cnts
    
    @classmethod
    def _draw_and_enumerate_contours(
            cls, 
            masked_3d_object_shape: tuple[int], 
            cnts: tuple[np.ndarray]) -> np.ndarray:
        """Method to draw and enumerate the internal contours of the 3d 
        printed object

        Parameters:
            masked_3d_object_shape (tuple[int]): 
                Shape of array dimensions
            cnts (tuple[np.ndarray]): 
                The tuple of the internal contours of the 3d printed object 
                to be drawn

        Returns:
            np.ndarray: 
                Image with the internal contours drawn and enumerated
        """
        
        infill_contours = np.zeros(
            masked_3d_object_shape, dtype=np.uint8)

        for (i, c) in enumerate(cnts):
            M = cv2.moments(c)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            cv2.drawContours(infill_contours, [c], -1, (255, 255, 255), 1)
            cv2.putText(
                infill_contours, 
                "{}".format(i+1), 
                (center_x-10, center_y+10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                1)
            
        return infill_contours
    
    @classmethod
    def _retrieve_contours_pixels_areas(
            cls, 
            cnts: tuple[np.ndarray]) -> List[List[object]]:
        """Method to calculate the internal 3d printed object contour areas

        Parameters:
            cnts (tuple[np.ndarray]): 
                The tuple of the internal contours of the 3d printed object

        Returns:
            List[List[object]]: 
                A list of lists with the contour areas in pixels and an index 
                to know which contour matches with the area
        """
        
        infill_pixels_areas = []

        for (i, c) in enumerate(cnts):
            infill_pixels_areas.append([i+1, cv2.contourArea(c)])
            
        return infill_pixels_areas
    
    @classmethod
    def _convert_areas_from_pixels_to_millimeters_squared(
            cls, 
            reference_object_width: float, 
            reference_object_pixels_area: float, 
            infill_pixels_areas: List[List[object]]) -> List[List[object]]:
        """Method to convert the areas in pixels to millimeters squared

        Parameters:
            reference_object_width (float): 
                Known real width of the reference object
            reference_object_pixels_area (float): 
                Reference object area in pixels
            infill_pixels_areas (List[List[object]]): 
                Internal 3d printed object contours areas in pixels

        Returns:
            List[List[object]]: 
                Internal 3d printed object contours areas in millimeters 
                squared
        """
        
        # Calculate area in mm2 of the reference object
        reference_object_area = math.pow(reference_object_width/2, 2) * math.pi
        
        infill_areas = []
        
        for i, infill_pixels_area in infill_pixels_areas:
            infill_areas.append([
                i, 
                infill_pixels_area 
                * reference_object_area 
                / reference_object_pixels_area])
            
        return infill_areas