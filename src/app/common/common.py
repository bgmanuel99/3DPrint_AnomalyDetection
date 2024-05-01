import cv2
import imutils
import numpy as np
from typing import List

def system_out(exception: Exception) -> None:
    print(exception)
    exit()
    
def print_image(
        image_name: str, 
        image: np.ndarray, 
        width=None, 
        show_flag=False) -> None:
    if show_flag:
        if width: 
            image = imutils.resize(image, width=width)
        else:
            image = imutils.resize(image, width=600)
        
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def print_images(
        image_names: List[str], 
        images: List[np.ndarray], 
        width = None, 
        show_flag=False) -> None:
    if show_flag:
        for image_name, image in zip(image_names, images):
            if width: 
                image = imutils.resize(image, width=width)
            else:
                image = imutils.resize(image, width=600)
            
            cv2.imshow(image_name, image)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()