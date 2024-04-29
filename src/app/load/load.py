import os
import sys
import cv2
import numpy

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.load_exceptions import *
from app.common.common import system_out
from app.utils.constants.constants import *

class Load(object):
    """Class for the process data load.

    Methods:
        load_process_data (image: numpy.ndarray): 
            Method to load process data.
        _check_directory:
            Private method to check if the output directory exists.
    """
    
    @classmethod
    def load_process_data(cls, image: numpy.ndarray):
        """Method to load process data.

        Parameters:
            image (numpy.ndarray): Final image with the detected anomalies
        """
        
        cls._check_directory()
        
        cv2.imwrite(
            "{}{}{}.{}".format(
                os.path.dirname(os.getcwd()), 
                output_image_directory_path, 
                "result", 
                output_image_file_extension), 
            image)
        
    
    @classmethod
    def _check_directory(cls):
        """Method to check if the output directory exists.

        Raises:
            OutputImageDirectoryNotFound: 
                Raised when the image output directory is not found
        """
        
        try:
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + output_image_directory_path): 
                    raise OutputImageDirectoryNotFound()
        except OutputImageDirectoryNotFound as e:
            system_out(e)