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