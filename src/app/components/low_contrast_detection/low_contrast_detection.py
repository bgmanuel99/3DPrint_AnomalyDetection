import os
import sys
import numpy
import cv2 as cv
from skimage.exposure import is_low_contrast

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.low_contrast_exceptions import *
from app.common.common import system_out

# TODO: Search for histogram equalization to enhance image contrast if it is too low
class LowContrastDetection(object):
    
    """This class contains the algorithms to detect low contrast images.
    
    Methods:
        low_contrast_dectection (image: numpy.ndarray):
            Determines if an image is low contrast.
    """
    
    @classmethod
    def low_contrast_dectection(cls, image: numpy.ndarray):
        """Determines if an image is low contrast.

        Parameters:
            image (numpy.ndarray): Image file

        Raises:
            LowContrastDetectionException: Raised when an image is low contrast and cannot be used in the pipeline
        """
        try:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            if is_low_contrast(image): raise LowContrastDetectionException()
        except LowContrastDetectionException as e:
            system_out(e)