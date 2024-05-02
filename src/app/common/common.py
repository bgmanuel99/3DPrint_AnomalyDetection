import cv2
import imutils
import numpy as np
from typing import List

class CommonPrints(object):
    
    """Class containing common functions of printing

    Methods:
        system_out (exception: Exception):
            Static method for printing exceptions and exiting from the 
            execution
        print_image (
                image_name: str, 
                image: np.ndarray, 
                width=None, 
                show_flag=False):
            Static method to print an image with opencv imshow function
        print_images (
                image_names: List[str], 
                images: List[np.ndarray], 
                width = None, 
                show_flag=False):
            Static method to print multiple images with opencv imshow function
    """
    
    @staticmethod
    def system_out(exception: Exception) -> None:
        """Static method for printing exceptions and exiting from the 
        execution

        Parameters:
            exception (Exception): 
                Exception given during execution to be printed out
        """
        
        print(exception)
        exit()
        
    @staticmethod    
    def print_image(
            image_name: str, 
            image: np.ndarray, 
            width: int=None, 
            show_flag: bool=False) -> None:
        """Static method to print an image with opencv imshow function

        Parameters:
            image_name (str): Image name
            image (np.ndarray): Image to print
            width (int, optional): Width of the image. Defaults to None.
            show_flag (bool, optional): 
                Flag to show or not the image. Defaults to False.
        """
        
        if show_flag:
            if width: 
                image = imutils.resize(image, width=width)
            else:
                image = imutils.resize(image, width=600)
            
            cv2.imshow(image_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    @staticmethod
    def print_images(
            image_names: List[str], 
            images: List[np.ndarray], 
            width: int=None, 
            show_flag: bool=False) -> None:
        """Static method to print multiple images with opencv imshow function

        Parameters:
            image_names (List[str]): Image name
            images (List[np.ndarray]): Image to print
            width (int, optional): Width of the image. Defaults to None.
            show_flag (bool, optional): 
                Flag to show or not the image. Defaults to False.
        """
        
        if show_flag:
            for image_name, image in zip(image_names, images):
                if width: 
                    image = imutils.resize(image, width=width)
                else:
                    image = imutils.resize(image, width=600)
                
                cv2.imshow(image_name, image)
                
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
            
        translated_image = np.zeros(original_shape, dtype=np.uint8)
        
        x_offset = round(translation_coords[0] + extra_offset)
        y_offset = round(translation_coords[1] + extra_offset)
        
        x_end = x_offset + external_contour_box.shape[1]
        y_end = y_offset + external_contour_box.shape[0]
        
        translated_image[
            y_offset:y_end, 
            x_offset:x_end] = external_contour_box
        
        return translated_image
    
class CommonMorphologyOperations(object):
    
    """Class contining common functions for morphology operations over images
    
    Methods:
        morphologyEx_opening (
                segmented_image: np.ndarray, 
                kernel_size: tuple[float, float]):
            Method to make an opening morphology operation over an image
    """
    
    @staticmethod
    def morphologyEx_opening(
            segmented_image: np.ndarray, 
            kernel_size: tuple[float, float]) -> np.ndarray:
        """Method to make an opening morphology operation over an image

        Parameters:
            segmented_image (np.ndarray): 
                Image to which the operation is performed
            kernel_size (tuple[float, float]): 
                Kernel size of the opening operation

        Returns:
            np.ndarray: Image with the opening operation
        """
        
        kernel = np.ones(kernel_size, np.uint8)
        
        opening = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
        
        return opening