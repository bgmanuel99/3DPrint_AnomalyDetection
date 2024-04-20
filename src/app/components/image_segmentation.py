import imutils
import cv2 as cv
import numpy as np
from typing import List
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

class ImageSegmetation(object):
    
    """This class contains methods to segmentated the original image with the
    3D printed and reference objects and to get a pixels per metric value 
    based on the reference object

    Methods:
        segment_image (image: np.ndarray): 
            Method to obtain the segmentation of the 3d printed object
        _get_complete_segmented_image(image: np.ndarray): 
            Private method to obtain the complete segmentation of the original 
            image
        _get_contours (segmented: np.ndarray): 
            Private method to obtain the contours of the objects of the 
            original image
        _get_3d_object_masked (
                image: np.ndarray, 
                segmented: np.ndarray): 
            Private method to obtain only the 3d printed object segmentation through its contour
        _mid_point (ptA: np.ndarray, ptB: np.ndarray):
            Private method to calculate the mid points of an edge
        get_3d_object_middle_coords ():
            Method to obtain the middle coordinates of the 3d object in the 
            original image
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
        """
        
        # Segment the original image
        segmented: np.ndarray = cls._get_complete_segmented_image(image)
        
        plt.imshow(segmented, cmap="gray")
        plt.show()
        
        # Obtain the contours of the object in the image
        cls._get_contours(segmented)
        
        # Obtain only the 3d object segmentation
        masked_object: np.ndarray = cls._get_3d_object_masked(image, segmented)
        
        return masked_object
    
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
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(gray_image, (7, 7), 0)
        
        _, segmented = cv.threshold(
            blurred_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        
        return segmented
    
    @classmethod
    def _get_contours(cls, segmented: np.ndarray) -> None:
        """Method to obtain the contours of the objects of the original image

        Parameters:
            segmented (np.ndarray): The segmentation of the original image
        """
        
        # Find contours in the segmented image
        cnts = cv.findContours(
            segmented, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Sort the contours from bottom to top in order to get the contour
        # of the reference object as the first variable
        (cnts, _) = contours.sort_contours(cnts, method="bottom-to-top")
        
        # TODO: Try to do this filter dynamic and remove static 1000
        # Eliminate any contour with an area lesser than a thousand
        cls._cnts = [c for c in cnts if cv.contourArea(c) > 1000]
    
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
        
        cv.drawContours(external_contour, [cls._cnts[1]], -1, (255, 255, 255), 2)
        
        plt.imshow(external_contour, cmap="gray")
        plt.show()
        
        cv.fillPoly(filled_contour, [cls._cnts[1]], (255, 255, 255))
        
        full_contour = external_contour + filled_contour
        
        # This first bitwise operation is to mask the 3d printed object
        # getting rid of the reference object of the original image
        masked = cv.bitwise_and(image, image, mask=full_contour)
        
        plt.imshow(masked, cmap="gray")
        plt.show()
        
        # This second bitwise operation is to separate the 3d printed object
        # forground from the background
        masked = cv.bitwise_and(masked, masked, mask=segmented)
        
        plt.imshow(masked, cmap="gray")
        plt.show()
        
        return masked
    
    @classmethod
    def _mid_point(cls, ptA: np.ndarray, ptB: np.ndarray) -> tuple[np.float64]:
        """Method to calculate the mid points of an edge

        Parameters:
            ptA (np.ndarray): First set of 2D coordinates
            ptB (np.ndarray): Second set of 2D coordinates

        Returns:
            tuple[np.float64]: Mid points
        """
        
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
    @classmethod
    def get_3d_object_middle_coords(cls) -> tuple[float, float]:
        """Method to obtain the middle coordinates of the 3d object in the 
        original image

        Returns:
            tuple[float, float]: First float is the X coordinate and the 
            second float is the Y coordinate
        """
        
        box = cv.minAreaRect(cls._cnts[1])
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (tl, tr, br, bl) = box
        
        (tlblX, tlblY) = cls._mid_point(tl, bl)
        (trbrX, trbrY) = cls._mid_point(tr, br)
        
        (midPointX, midPointY) = cls._mid_point((tlblX, tlblY), (trbrX, trbrY))
        
        return (midPointX, midPointY)
    
    @classmethod
    def get_pixels_per_metric(cls) -> float:
        """Method to obtain the pixels per metric variable based on a 
        reference object in the original image

        Returns:
            float: Pixels per metric value given by the width of the reference 
            object
        """
        
        box = cv.minAreaRect(cls._cnts[0])
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (tl, tr, br, bl) = box
        (tlblX, tlblY) = cls._mid_point(tl, bl)
        (trbrX, trbrY) = cls._mid_point(tr, br)
        
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        return dB