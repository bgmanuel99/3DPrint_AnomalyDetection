import os
import sys
import cv2
import imutils
import numpy as np
from typing import List
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.common.common import (
    CommonPrints, 
    CommonFunctionalities, 
    CommonMorphologyOperations)

class ImageSegmetation(object):
    """This class contains methods to segmentated the original image with the
    3D printed and reference objects and to get a pixels per metric value 
    based on the reference object

    Methods:
        segment_image (image: np.ndarray): 
            Method to obtain the segmentation of the 3d printed object
        _get_complete_segmented_image (image: np.ndarray): 
            Private method to obtain the complete segmentation of the original 
            image
        _get_contours (segmented: np.ndarray): 
            Private method to obtain the contours of the objects of the 
            original image
        _get_3d_object_masked (
                image: np.ndarray, 
                segmented: np.ndarray): 
            Private method to obtain only the 3d printed object segmentation 
            through its contour
        _get_3d_object_data (
                printed_object_box_coordinates: tuple[tuple[float]]):
            Private method to obtain all neccesary data from the contour of 
            the 3d printed object
        _transform_masked_3d_object( 
                masked_object: np.ndarray, 
                top_left_coord_3d_object: tuple[float], 
                printed_object_box_coordinates: tuple[tuple[float]]):
            Private method to get the perspective of the 3d printed object
        _mid_point (
                point_A: tuple[float], 
                point_B: tuple[float]):
            Private method to calculate the mid points of an edge
        _get_pixels_per_metric ():
            Private method to obtain the a list of pixels per metric values 
            variations representing degree offsets when taking the picture of 
            the original image
    """
    
    _cnts: List[np.ndarray] = None
    
    @classmethod
    def segment_image(cls, image: np.ndarray) -> tuple[
            np.ndarray, float, tuple[float], tuple[float]]:
        """Method to obtain the segmentation of the 3d printed object

        Parameters:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Segmented image
            float: Variable with pixels per metric value
            tuple[float]: Middle coordinates of the 3d printed object
            tuple[float]: Top left coordinates of the 3d printed object
        """
        
        # Segment the original image
        segmented: np.ndarray = cls._get_complete_segmented_image(image)
        
        CommonPrints.print_image("segmented", segmented, 600)
        
        # Obtain the contours of the object in the image
        cls._get_contours(segmented)
        
        # Obtain only the 3d object segmentation
        masked_object: np.ndarray = cls._get_3d_object_masked(image, segmented)
        
        printed_object_box_coordinates = CommonFunctionalities \
            .get_box_coordinates(cls._cnts[1])
        
        # Calculate middle coordinates of the 3d printed object in the original
        # image
        (middle_coords_3d_object, 
         top_left_coord_3d_object) = cls._get_3d_object_data(
            printed_object_box_coordinates)
        
        # Transform the object to eliminate distorsion
        transformed_object: np.ndarray = cls \
            ._transform_and_translate_masked_3d_object(
                masked_object, 
                top_left_coord_3d_object, 
                printed_object_box_coordinates)
                
        return (transformed_object, 
                cls._get_pixels_per_metric(), 
                middle_coords_3d_object, 
                top_left_coord_3d_object)
    
    @classmethod
    def _get_complete_segmented_image(cls, image: np.ndarray) -> np.ndarray:
        """Method to obtain the complete segmentation of the original image

        Parameters:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Complete segmentation of the original image
        """
        
        segmented_image = CommonFunctionalities.get_segmented_image(image)
        
        CommonPrints.print_image("segmented", segmented_image, 600)
        
        opening = CommonMorphologyOperations.morphologyEx_opening(
            segmented_image, (5, 5))
        
        return opening
    
    @classmethod
    def _get_contours(cls, segmented: np.ndarray) -> None:
        """Method to obtain the contours of the objects of the original image

        Parameters:
            segmented (np.ndarray): The segmentation of the original image
        """
        
        # Find contours in the segmented image
        cnts = CommonFunctionalities.find_and_grab_contours(segmented)
        
        # Sort the contours from bottom to top in order to get the contour
        # of the reference object as the first variable
        (cnts, _) = contours.sort_contours(cnts, method="bottom-to-top")
        
        # TODO: Try to do this filter dynamic and remove static 1000
        # Eliminate any contour with an area lesser than a thousand
        cls._cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
    
    @classmethod
    def _get_3d_object_masked(
            cls, 
            image: np.ndarray, 
            segmented: np.ndarray) -> np.ndarray:
        """Method to obtain only the 3d printed object segmentation through
        its contour

        Parameters:
            image (np.ndarray): Original image
            segmented (np.ndarray): Complete segmentation of the original image

        Returns:
            np.ndarray: Segmentation of the 3d printed object
        """
        
        # Create a mask only for the 3d printed object based on the contour
        filled_contour = np.zeros(image.shape[0:2], dtype=np.uint8)
        
        cv2.fillPoly(filled_contour, [cls._cnts[1]], (255, 255, 255))
        
        # This first bitwise operation is to mask the 3d printed object
        # getting rid of the reference object of the original image
        masked = cv2.bitwise_and(image, image, mask=filled_contour)
        
        CommonPrints.print_image("masked", masked, 600)
        
        # This second bitwise operation is to separate the 3d printed object
        # forground from the background
        masked = cv2.bitwise_and(masked, masked, mask=segmented)
        
        CommonPrints.print_image("masked", masked, 600)
        
        return masked
    
    @classmethod
    def _get_3d_object_data(
            cls, 
            printed_object_box_coordinates: tuple[tuple[float]]
            ) -> tuple[tuple[float], tuple[float]]:
        """Method to obtain all neccesary data from the contour of the 3d 
        printed object
        
        Parameters:
            printed_object_box_coordinates (tuple[tuple[float]]): 
                The box coordinates of a contour
        
        Returns:
            tuple[float]: Middle coordinates of the 3d printed object
            tuple[float]: Top left coordinates of the 3d printed object
        """
        
        (top_left, top_right, bottom_right, bottom_left) = perspective \
            .order_points(printed_object_box_coordinates)
 
        printed_object_left_mid_point = cls._mid_point(
            top_left, bottom_left)
        printed_object_right_mid_point = cls._mid_point(
            top_right, bottom_right)
        
        # Compute the middle coordinates of the 3d printed object
        middle_coords_3d_object = cls._mid_point(
            printed_object_left_mid_point, printed_object_right_mid_point)
        
        return middle_coords_3d_object, top_left
    
    @classmethod
    def _transform_and_translate_masked_3d_object(
            cls, 
            masked_object: np.ndarray, 
            top_left_coord_3d_object: tuple[float], 
            printed_object_box_coordinates: tuple[tuple[float]]) -> np.ndarray:
        """Method to get the perspective of the 3d printed object

        Parameters:
            masked_object (np.ndarray): 
                Segmentation of the 3d printed object
            top_left_coord_3d_object (tuple[float]): 
                Top left coordinates of the 3d printed object in the original
                image
            printed_object_box_coordinates (tuple[tuple[float]]): 
                The box coordinates of a contour

        Returns:
            np.ndarray: 3d printed object with the perspective transformation
        """
        
        warped = perspective.four_point_transform(
            masked_object, printed_object_box_coordinates)

        CommonPrints.print_image("warped", warped, 600)
        
        segmented_image = CommonFunctionalities.get_segmented_image(warped)
        
        cnts = CommonFunctionalities.find_and_grab_contours(segmented_image)
        
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
        
        box = CommonFunctionalities.get_box_coordinates(cnts[0])
        (top_left, top_right, bottom_right, bottom_left) = perspective \
            .order_points(box)
        
        translated_object = CommonFunctionalities.get_translated_object(
            warped, 
            top_left,
            top_right, 
            bottom_right, 
            bottom_left, 
            top_left_coord_3d_object, 
            masked_object.shape, 
            0)
        
        CommonPrints.print_image("transformed_image", translated_object, 600)
        
        return translated_object
    
    @classmethod
    def _mid_point(
            cls, 
            point_A: tuple[float], 
            point_B: tuple[float]) -> tuple[np.float64]:
        """Method to calculate the mid points of an edge

        Parameters:
            point_A (tuple[float]): First set of 2D coordinates
            point_B (tuple[float]): Second set of 2D coordinates

        Returns:
            tuple[np.float64]: Mid points
        """
        
        return (
            (point_A[0] + point_B[0]) * 0.5, 
            (point_A[1] + point_B[1]) * 0.5)
    
    @classmethod
    def _get_pixels_per_metric(cls) -> tuple[float]:
        """Method to obtain the a list of pixels per metric values variations 
        representing degree offsets when taking the picture of the original 
        image

        Returns:
            List[float]: 
                List of pixels per metric values variations 
                representing degree offsets when taking the picture of the 
                original image
        """
        
        box = CommonFunctionalities.get_box_coordinates(cls._cnts[0])
        (top_left, top_right, bottom_right, bottom_left) = perspective \
            .order_points(box)
            
        printed_object_left_mid_point = cls._mid_point(
            top_left, bottom_left)
        printed_object_right_mid_point = cls._mid_point(
            top_right, bottom_right)
        
        width = dist.euclidean(
            printed_object_left_mid_point, printed_object_right_mid_point)
        
        ppm_degree_offset = []
        
        for offset in [i * 0.1 for i in range(-10, 11)]:
            ppm_degree_offset.append(width + offset)
        
        return ppm_degree_offset