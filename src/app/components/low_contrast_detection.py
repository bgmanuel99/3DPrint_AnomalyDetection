import cv2
import numpy as np
from skimage.exposure import is_low_contrast

from app.utils.constants.constants import *

class LowContrastDetection(object):
    """This class contains the algorithm to detect low contrast images.
    
    Methods:
        low_contrast_detection (image: numpy.ndarray):
            Method to determines if an image is low contrast.
            
    Raises:
        LowContrastDetectionException: 
            Raised when an image is low contrast and cannot be used in the 
            pipeline
    """
    
    @classmethod
    def low_contrast_detection(cls, image: np.ndarray) -> np.ndarray | None:
        """Method which determines if an image is low contrast and for the 
        adaptive histogram equalization in the case it is.

        Parameters:
            image (numpy.ndarray): Image file

        Raises:
            LowContrastDetectionException: 
                Raised when an image is low contrast and cannot be used in the 
                pipeline
        """
        
        print("[INFO] Low contrast process")
        
        if is_low_contrast(image, fraction_threshold=FRACTION_THRESHOLD): 
            gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            return clahe.apply(gray)
        else: 
            return image