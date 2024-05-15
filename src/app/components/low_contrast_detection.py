import cv2
import numpy
from skimage.exposure import is_low_contrast

from app.utils.constants.constants import *
from app.common.common_prints import CommonPrints
from app.utils.exceptions.low_contrast_exceptions import *

# TODO: Search for histogram equalization to enhance image contrast if it is 
# too low
class LowContrastDetection(object):
    """This class contains the algorithms to detect low contrast images.
    
    Methods:
        low_contrast_dectection (image: numpy.ndarray):
            Method to determines if an image is low contrast.
            
    Raises:
        LowContrastDetectionException: 
            Raised when an image is low contrast and cannot be used in the 
            pipeline
    """
    
    @classmethod
    def low_contrast_dectection(cls, image: numpy.ndarray):
        """Method to determines if an image is low contrast.

        Parameters:
            image (numpy.ndarray): Image file

        Raises:
            LowContrastDetectionException: 
                Raised when an image is low contrast and cannot be used in the 
                pipeline
        """
        
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if is_low_contrast(image, fraction_threshold=FRACTION_THRESHOLD): 
                raise LowContrastDetectionException()
        except LowContrastDetectionException as e:
            CommonPrints.system_out(e)