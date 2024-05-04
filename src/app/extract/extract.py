import os
import io
import sys
import cv2
import numpy as np

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.extract_exceptions import *
from app.common.common import CommonPrints
from app.utils.constants.constants import *

class Extract(object):
    """Class for the process data extraction.
    
    Methods:
        extract_process_data (
                gcode_name: str, 
                image_name: str, 
                metadata_name: str): 
            Method to extract process data.
        _check_directories: 
            Private method to check if the directories of the input files
            exists.
        _check_data (
                gcode_path: str,
                gcode_name: str, 
                image_path: str,
                image_name: str, 
                metadata_path: str, 
                metadata_name: str): 
            Private method to check if the input files exists.
    """
    
    @classmethod
    def extract_process_data(
            cls, 
            gcode_name: str, 
            image_name: str, 
            # TODO: Terminar de añadir devolucion del metadato al metodo
            metadata_name: str) -> tuple[io.TextIOWrapper, np.ndarray]:
        """Method to extract process data.

        Parameters:
            gcode_name (str): Name of the gcode file
            image_name (str): Name of the image
            metadata_name (str): Name of the metadata file

        Returns:
            # TODO: Terminar de añadir devolucion del metadato al metodo
            tuple[io.TextIOWrapper, numpy.ndarray]
                Gcode file, image and metadata file
        """
        
        gcode_path = "{}{}{}.{}".format(
            os.path.dirname(os.getcwd()), 
            input_gcode_directory_path, 
            gcode_name, 
            gcode_file_extension)
        
        image_path = "{}{}{}.{}".format(
            os.path.dirname(os.getcwd()), 
            input_image_directory_path, 
            image_name, 
            image_file_extension)
        
        metadata_path = "{}{}{}.{}".format(
            os.path.dirname(os.getcwd()), 
            input_metadata_directory_path, 
            metadata_name, 
            metadata_file_extension
        )
        
        # Check if directories exists
        cls._check_directories()
        
        # Check if data exists and are files
        cls._check_data(
            gcode_path, 
            gcode_name, 
            image_path, 
            image_name, 
            metadata_path, 
            metadata_name)
        
        # Extract data
        gcode_file = open(gcode_path, "r")
        
        image = cv2.imread(image_path)
        
        metadata_file = open(metadata_path, "r")
        
        return gcode_file, image, metadata_file
    
    @classmethod
    def _check_directories(cls):
        """Method to check if the directories of the input files exists.

        Raises:
            GCodeDirectoryNotFoundException: 
                Raised when the gcode input directory is not found
            ImageDirectoryNotFoundException: 
                Raised when the image input directory is not found
        """
        
        try:
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + input_gcode_directory_path): 
                    raise InputGCodeDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + input_image_directory_path): 
                    raise InputImageDirectoryNotFoundException()
        except (
            InputGCodeDirectoryNotFoundException, InputImageDirectoryNotFoundException) as e:
            CommonPrints.system_out(e)
    
    @classmethod
    def _check_data(
            cls, 
            gcode_path: str, 
            gcode_name: str,
            image_path: str,
            image_name: str):
        """Method to check if the input files exists.

        Parameters:
            gcode_path (str): Path to the gcode file
            gcode_name (str): Name of the gcode file
            image_path (str): Path to the image
            image_name (str): Name of the image

        Raises:
            GCodeNotFileException: 
                Raised when the input gcode is not a file
            ExtractGCodeFileException: 
                Raised when the input gcode file cannot be found
            ImageNotFileException: 
                Raised when the input image is not a file
            ExtractImageException: 
                Raised when the input image cannot be found
        """
        
        try:
            if os.path.exists(gcode_path):
                if not os.path.isfile(gcode_path):
                    raise GCodeNotFileException(gcode_name)
            else: 
                raise ExtractGCodeFileException(gcode_name)
            
            if os.path.exists(image_path):
                if not os.path.isfile(image_path):
                    raise ImageNotFileException(image_name)
            else: 
                raise ExtractImageException(image_name)
        except (
            GCodeNotFileException, 
            ExtractGCodeFileException, 
            ImageNotFileException, 
            ExtractImageException) as e:
            CommonPrints.system_out(e)