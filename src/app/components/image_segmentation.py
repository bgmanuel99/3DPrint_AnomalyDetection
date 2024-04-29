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

from app.common.common import print_image

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
        _transform_masked_3d_object( 
                masked_object: np.ndarray, 
                top_left_coord_3d_object: tuple[float]):
            Private method to get the perspective of the 3d printed object
        _mid_point (
                point_A: tuple[float], 
                point_B: tuple[float]):
            Private method to calculate the mid points of an edge
        _get_3d_object_data ():
            Private method to obtain all neccesary data from the contour of 
            the 3d printed object
        get_pixels_per_metric ():
            Method to obtain the pixels per metric variable based on a 
            reference object in the original image
    """
    
    _cnts: List[np.ndarray] = None
    
    @classmethod
    def segment_image(cls, image: np.ndarray) -> np.ndarray:
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
        
        print_image("segmented", segmented, 600)
        
        # Obtain the contours of the object in the image
        cls._get_contours(segmented)
        
        # Obtain only the 3d object segmentation
        masked_object: np.ndarray = cls._get_3d_object_masked(image, segmented)
        
        # Calculate middle coordinates of the 3d printed object in the original
        # image
        (middle_coords_3d_object, 
         top_left_coord_3d_object) = cls._get_3d_object_data()
        
        # Transform the object to eliminate distorsion
        transformed_object: np.ndarray = cls._transform_masked_3d_object(
            masked_object, top_left_coord_3d_object)
                
        return (transformed_object, 
                cls.get_pixels_per_metric(), 
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
        
        # Convert the image to grayscale and blur it slightly
        # Blurring the image helps remove some of the high frequency edges 
        # in the image and allow a more clean segmentation
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        
        segmented = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        print_image("segmented", segmented, 600)
        
        kernel = np.ones((5, 5), np.uint8)
        
        opening = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    @classmethod
    def _get_contours(cls, segmented: np.ndarray) -> None:
        """Method to obtain the contours of the objects of the original image

        Parameters:
            segmented (np.ndarray): The segmentation of the original image
        """
        
        # Find contours in the segmented image
        cnts = cv2.findContours(
            segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
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
        external_contour = np.zeros(image.shape[0:2], dtype=np.uint8)
        filled_contour = external_contour.copy()
        
        cv2.drawContours(
            external_contour, [cls._cnts[1]], -1, (255, 255, 255), 2)
        
        cv2.fillPoly(filled_contour, [cls._cnts[1]], (255, 255, 255))
        
        full_contour = external_contour + filled_contour
        
        # This first bitwise operation is to mask the 3d printed object
        # getting rid of the reference object of the original image
        masked = cv2.bitwise_and(image, image, mask=full_contour)
        
        print_image("masked", masked, 600)
        
        # This second bitwise operation is to separate the 3d printed object
        # forground from the background
        masked = cv2.bitwise_and(masked, masked, mask=segmented)
        
        print_image("masked", masked, 600)
        
        return masked
    
    @classmethod
    def _transform_masked_3d_object(
            cls, 
            masked_object: np.ndarray, 
            top_left_coord_3d_object: tuple[float]):
        """Method to get the perspective of the 3d printed object

        Parameters:
            masked_object (np.ndarray): 
                Segmentation of the 3d printed object
            top_left_coord_3d_object (tuple[float]): 
                Top left coordinates of the 3d printed object in the original
                image

        Returns:
            np.ndarray: 3d printed object with the perspective transformation
        """
        
        box = cv2.minAreaRect(cls._cnts[1])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        warped = perspective.four_point_transform(masked_object, box)
        
        print(warped.shape)

        print_image("warped", warped, 600)
        
        gray_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        
        segmented = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        cnts = cv2.findContours(
            segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
        
        print("contours")
        print(len(cnts))
        
        box = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (top_left, top_right, bottom_right, bottom_left) = box
        
        print(top_left, top_right, bottom_right, bottom_left)
        
        top_left = list(map(int, top_left))
        top_right = list(map(int, top_right))
        bottom_right = list(map(int, bottom_right))
        bottom_left = list(map(int, bottom_left))

        warped_external_contour = warped[
            max(top_left[1], top_right[1]):
                max(bottom_right[1], bottom_left[1]), 
            max(top_left[0], bottom_left[0]):
                max(top_right[0], bottom_right[0])]
        
        print("shape")
        print(warped_external_contour.shape)
        
        print_image("warped_external_contour", warped_external_contour, 600)
            
        transformed_image = np.zeros(masked_object.shape, dtype=np.uint8)
        
        x_offset = round(top_left_coord_3d_object[0])
        y_offset = round(top_left_coord_3d_object[1])
        
        x_end = x_offset + warped_external_contour.shape[1]
        y_end = y_offset + warped_external_contour.shape[0]
        
        transformed_image[
            y_offset:y_end, 
            x_offset:x_end] = warped_external_contour
        
        print_image("transformed_image", transformed_image, 600)
        
        return transformed_image
    
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
    def _get_3d_object_data(cls):
        """Method to obtain all neccesary data from the contour of the 3d 
        printed object
        
        Returns:
            tuple[float]: Middle coordinates of the 3d printed object
            tuple[float]: Top left coordinates of the 3d printed object
        """
        
        box = cv2.minAreaRect(cls._cnts[1])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (top_left, top_right, bottom_right, bottom_left) = box
        print("box coords")
        print(top_left, top_right, bottom_right, bottom_left)
 
        (top_left_bottom_left_X, top_left_bottom_left_Y) = cls._mid_point(
            top_left, bottom_left)
        (top_right_bottom_right_X, top_right_bottom_right_Y) = cls._mid_point(
            top_right, bottom_right)
        
        # Compute the middle coordinates of the 3d printed object
        middle_coords_3d_object = cls._mid_point(
            (top_left_bottom_left_X, top_left_bottom_left_Y), 
            (top_right_bottom_right_X, top_right_bottom_right_Y))
        
        return middle_coords_3d_object, top_left
    
    @classmethod
    def get_pixels_per_metric(cls) -> float:
        """Method to obtain the pixels per metric variable based on a 
        reference object in the original image

        Returns:
            float: 
                Pixels per metric value given by the width of the reference 
                object
        """
        
        box = cv2.minAreaRect(cls._cnts[0])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (top_left, tr, bottom_right, bottom_left) = box
        (top_left_bottom_left_X, top_left_bottom_left_Y) = cls._mid_point(
            top_left, bottom_left)
        (top_right_bottom_right_X, top_right_bottom_right_Y) = cls._mid_point(
            tr, bottom_right)
        
        width = dist.euclidean(
            (top_left_bottom_left_X, top_left_bottom_left_Y), (top_right_bottom_right_X, top_right_bottom_right_Y))
        
        return width