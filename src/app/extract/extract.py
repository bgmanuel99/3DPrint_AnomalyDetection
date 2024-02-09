import os
import io
import sys
import numpy
import cv2 as cv
from typing import Tuple

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.extract_exceptions import *
from app.common.common import system_out
from app.utils.constants.constants import *

class Extract(object):
    
    """Class for the process data extraction.
    
    Methods:
        extract_process_data (file_name: str, image_name: str): 
            Method to extract process data.
        _check_directories: 
            Private method to check if the directories of the input files exists.
        _check_data (file_name: str, image_name: str): 
            Private method to check if the input files exists.
    """
    
    @classmethod
    def extract_process_data(cls, file_name: str, image_name: str) -> Tuple[io.TextIOWrapper, numpy.ndarray]:
        """Method to extract process data.

        Parameters:
            file_name (str): Name of the gcode file
            image_name (str): Name of the image

        Returns:
            io.TextIOWrapper: Gcode file
            numpy.ndarray: Image file
        """
        
        # Check if directories exists
        cls._check_directories()
        
        # Check if data exists
        cls._check_data(file_name, image_name)
        
        # Extract data
        file = open("{}{}{}.{}".format(os.path.dirname(os.getcwd()), gcode_directory_path, file_name, gcode_file_extension), "r")
        
        image = cv.imread("{}{}{}.{}".format(os.path.dirname(os.getcwd()), image_directory_path, image_name, image_file_extension))
        
        return file, image
    
    @classmethod
    def _check_directories(cls):
        """Method to check if the directories of the input files exists.

        Raises:
            GCodeDirectoryNotFoundException: Raised when the gcode input directory is not found
            ImageDirectoryNotFoundException: Raised when the image input directory is not found
        """
        
        try:
            if not os.path.exists(os.path.dirname(os.getcwd()) + gcode_directory_path): raise GCodeDirectoryNotFoundException()
            if not os.path.exists(os.path.dirname(os.getcwd()) + image_directory_path): raise ImageDirectoryNotFoundException()
        except GCodeDirectoryNotFoundException as e:
            system_out(e)
        except ImageDirectoryNotFoundException as e:
            system_out(e)
    
    @classmethod
    def _check_data(cls, file_name: str, image_name: str):
        """Method to check if the input files exists.

        Parameters:
            file_name (str): Name of the gcode file
            image_name (str): Name of the image

        Raises:
            GCodeNotFileException: Raised when the input gcode is not a file
            ExtractGCodeFileException: Raised when the input gcode file cannot be found
            ImageNotFileException: Raised when the input image is not a file
            ExtractImageException: Raised when the input image cannot be found
        """
        
        try:
            if os.path.exists("{}{}{}.{}".format(os.path.dirname(os.getcwd()), gcode_directory_path, file_name, gcode_file_extension)):
                if not os.path.isfile("{}{}{}.{}".format(os.path.dirname(os.getcwd()), gcode_directory_path, file_name, gcode_file_extension)):
                    raise GCodeNotFileException(file_name)
            else: raise ExtractGCodeFileException(file_name)
            
            if os.path.exists("{}{}{}.{}".format(os.path.dirname(os.getcwd()), image_directory_path, image_name, image_file_extension)):
                if not os.path.isfile("{}{}{}.{}".format(os.path.dirname(os.getcwd()), image_directory_path, image_name, image_file_extension)):
                    raise ImageNotFileException(image_name)
            else: raise ExtractImageException(image_name)
        except GCodeNotFileException as e:
            system_out(e)
        except ExtractGCodeFileException as e:
            system_out(e)
        except ImageNotFileException as e:
            system_out(e)
        except ExtractImageException as e:
            system_out(e)