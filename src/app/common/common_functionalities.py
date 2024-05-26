import cv2
import imutils
import numpy as np
from tqdm import tqdm
from typing import List
from skimage.metrics import structural_similarity

from app.common.common_prints import CommonPrints

class CommonFunctionalities(object):
    """Class containing common functionalities of the modules

    Methods:
        get_box_coordinates (contour: np.ndarray):
            Method to obtain the minimun area of a contour as a rectangle with
            fore coordinates
        get_segmented_image (image: np.ndarray):
            Method to obtain a segmented image
        find_and_grab_contours (
                segmented_image: np.ndarray):
            Method to find the contours of a segmented image and grab them
        get_translated_object (
                image: np.ndarray, 
                top_left: tuple[float], 
                top_right: tuple[float], 
                bottom_right: tuple[float], 
                bottom_left: tuple[float], 
                translation_coords: tuple[float], 
                original_shape: tuple[float], 
                extra_offset: int):
            Method to translate and object to another part of the image
        calculate_ssim_max_score (
                segmented_object: np.ndarray, 
                perfect_models: List[np.ndarray]):
            Method to calculate the max SSIM score between a segmented image 
            and its perfect models
    """
    
    @staticmethod
    def get_box_coordinates(contour: np.ndarray) -> np.ndarray:
        """Method to obtain the minimun area of a contour as a rectangle with
        fore coordinates

        Parameters:
            contour (np.ndarray): The contour of an object

        Returns:
            np.ndarray: The box coordinates of the contour
        """
        
        # Gets the minimum area of the contour as a rectangle with fore coords
        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        
        return box
    
    @staticmethod
    def get_segmented_image(image: np.ndarray) -> np.ndarray:
        """Method to obtain a segmented image

        Parameters:
            image (np.ndarray): Image to be segmented

        Returns:
            np.ndarray: Segmented image
        """
        
        # Convert the image to grayscale and blur it slightly
        # Blurring the image helps remove some of the high frequency edges 
        # in the image and allow a more clean segmentation
        if image.shape[2] == 1:
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        
        segmented_image = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        return segmented_image
    
    @staticmethod
    def find_and_grab_contours(
            segmented_image: np.ndarray) -> tuple[np.ndarray]:
        """Method to find the contours of a segmented image and grab them

        Parameters:
            segmented_image (np.ndarray): Segmented image

        Returns:
            tuple[np.ndarray]: Contours of the segmented image
        """
        
        cnts = cv2.findContours(
            segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        return cnts
    
    @staticmethod
    def get_translated_object(
            image: np.ndarray, 
            top_left: tuple[float], 
            top_right: tuple[float], 
            bottom_right: tuple[float], 
            bottom_left: tuple[float], 
            translation_coords: tuple[float], 
            original_shape: tuple[float], 
            extra_offset: int) -> np.ndarray:
        """Method to translate and object to another part of the image

        Parameters:
            image (np.ndarray): Image to be translated
            top_left (tuple[float]): Top left coordinate of the image
            top_right (tuple[float]): Top right coordinate of the image
            bottom_right (tuple[float]): Bottom right coordinate of the image
            bottom_left (tuple[float]): Bottom left coordinate of the image
            translation_coords (tuple[float]): 
                New origin coordinates to translate the image
            original_shape (tuple[float]): 
                Original shape of the an initial image in which the new passed 
                image will be positioned
            extra_offset (int): 
                Extra offset to add to the translation coordinates if neccesary

        Returns:
            np.ndarray: 
                A new image with the one passed to the function in the
                position of the translation coordinates plus extra offset if 
                neccesary
        """
        
        (top_left, top_right, bottom_right, bottom_left) = tuple(map(
            lambda sublist: tuple(map(int, sublist)), 
            (top_left, top_right, bottom_right, bottom_left)))

        external_contour_box = image[
            max(top_left[1], top_right[1]):
                max(bottom_right[1], bottom_left[1]), 
            max(top_left[0], bottom_left[0]):
                max(top_right[0], bottom_right[0])]
        
        CommonPrints.print_image("external", external_contour_box, 600)
            
        translated_image = np.zeros(original_shape, dtype=np.uint8)
        
        x_offset = round(translation_coords[0] + extra_offset)
        y_offset = round(translation_coords[1] + extra_offset)
        
        x_end = x_offset + external_contour_box.shape[1]
        y_end = y_offset + external_contour_box.shape[0]
        
        translated_image[
            y_offset:y_end, 
            x_offset:x_end] = external_contour_box
        
        return translated_image
    
    @classmethod
    def calculate_ssim_max_score(
            cls, 
            segmented_object: np.ndarray, 
            perfect_models: List[np.ndarray]) -> tuple[float, int]:
        """Method to calculate the max SSIM score between a segmented image 
        and its perfect models

        Parameters:
            segmented_object (np.ndarray): 
                Image of the segmented object
            perfect_models (List[np.ndarray]): 
                List of perfect models

        Returns:
            tuple[float, int]: 
                - SSIM max score
                - Index to know which perfect model gave the best SSIM score
        """
        
        ssim_max_score = 0
        ssim_max_score_index = 0
        
        for i in tqdm(range(len(perfect_models))):
            ssim_score = structural_similarity(
                perfect_models[i], segmented_object, full=True)[0]
            
            if ssim_score > ssim_max_score:
                ssim_max_score = ssim_score
                ssim_max_score_index = i
        
        return ssim_max_score, ssim_max_score_index