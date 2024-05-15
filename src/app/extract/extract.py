import os
import io
import cv2
import imutils
import numpy as np

from app.utils.constants.constants import *
from app.common.common_prints import CommonPrints
from app.utils.exceptions.extract_exceptions.input_image_exceptions import *
from app.utils.exceptions.extract_exceptions.input_gcode_exceptions import *
from app.utils.exceptions.extract_exceptions.input_metadata_exceptions import *

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
            Private method to check if the input files exists and are files.
            
    Raises:
        InputGCodeDirectoryNotFoundException: 
            Raised when the gcode input directory is not found
        InputImageDirectoryNotFoundException: 
            Raised when the image input directory is not found
        InputMetadataDirectoryNotFoundException:
            Raised when the metadata input directory is not found
        ExtractImageException: 
            Raised when the input image cannot be found
        ImageNotFileException: 
            Raised when the input image is not a file
        NonSupportedImageExtensionException:
            Raised when the input image has a non supported file extension
        ExtractGCodeFileException: 
            Raised when the input gcode file cannot be found
        GCodeNotFileException: 
            Raised when the input gcode is not a file
        NonSupportedGcodeExtensionException:
            Raised when the input gcode file has a non supported file 
            extension
        ExtractMetadataException:
            Raised when the input metadata file cannot be found
        MetadataNotFileException:
            Raised when the input metadata is not a file
        NonSupportedMetadataExtensionException:
            Raised when the input metadata file has a non supported file 
            extension
    """
    
    @classmethod
    def extract_process_data(
            cls, 
            gcode_name: str, 
            image_name: str, 
            metadata_name: str) -> (
                tuple[io.TextIOWrapper, np.ndarray, io.TextIOWrapper] 
                | tuple[io.TextIOWrapper, np.ndarray, str]
            ):
        """Method to extract process data.

        Parameters:
            gcode_name (str): Name of the gcode file
            image_name (str): Name of the image
            metadata_name (str): Name of the metadata file

        Returns:
            (
                tuple[io.TextIOWrapper, numpy.ndarray, io.TextIOWrapper] 
                | tuple[io.TextIOWrapper, np.ndarray, str]
            ):
                Gcode file, image and metadata file or Gcode file, image and 
                metadata name that should equals a void string
        """
        
        gcode_path = "{}{}{}{}".format(
            os.path.dirname(os.getcwd()), 
            INPUTS_FATHER_DIRECTORY_PATH, 
            INPUT_GCODE_DIRECTORY, 
            gcode_name)
        
        image_path = "{}{}{}{}".format(
            os.path.dirname(os.getcwd()), 
            INPUTS_FATHER_DIRECTORY_PATH, 
            INPUT_IMAGE_DIRECTORY, 
            image_name)
        
        metadata_path = "{}{}{}{}".format(
            os.path.dirname(os.getcwd()), 
            INPUTS_FATHER_DIRECTORY_PATH, 
            INPUT_METADATA_DIRECTORY, 
            metadata_name)
        
        # trainX = []
        # for image_name in sorted([
        #     int(name.split(".")[0]) for name in os.listdir(
        #     os.path.dirname(os.getcwd()) + "/data/classification/trainX/")]):
        #     image = cv2.imread(
        #         "{}{}{}.jpg".format(
        #             os.path.dirname(os.getcwd()), 
        #             "/data/classification/trainX/", 
        #             image_name))
        #     image = imutils.resize(image, width=120)
        #     trainX.append(image)
        # trainX = np.array(trainX)
        
        # trainY = []
        # trainY_file = open(
        #     "{}{}".format(
        #         os.path.dirname(os.getcwd()), 
        #         "/data/classification/trainY/trainY.txt"), 
        #     "r")
        # for label in trainY_file.readlines():
        #     label = label.strip().replace("\n", "")
        #     trainY.append(label)
        # trainY = np.array(trainY)
        # trainY = trainY.astype(np.uint8)
        
        # testX = []
        # for image_name in sorted([
        #     int(name.split(".")[0]) for name in os.listdir(
        #     os.path.dirname(os.getcwd()) + "/data/classification/testX/")]):
        #     image = cv2.imread(
        #         "{}{}{}.jpg".format(
        #             os.path.dirname(os.getcwd()), 
        #             "/data/classification/testX/", 
        #             image_name))
        #     image = imutils.resize(image, width=120)
        #     testX.append(image)
        # testX = np.array(testX)
        
        # testY = []
        # testY_file = open(
        #     "{}{}".format(
        #         os.path.dirname(os.getcwd()), 
        #         "/data/classification/testY/testY.txt"), 
        #     "r")
        # for label in testY_file.readlines():
        #     label = label.strip().replace("\n", "")
        #     testY.append(label)
        # testY = np.array(testY)
        # testY = testY.astype(np.uint8)
        
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
        
        if not metadata_name == "":
            metadata_file = open(metadata_path, "r")
            
            return gcode_file, image, metadata_file
        else:
            return gcode_file, image, metadata_name
    
    @classmethod
    def _check_directories(cls):
        """Method to check if the directories of the input files exists.

        Raises:
            InputGCodeDirectoryNotFoundException: 
                Raised when the gcode input directory is not found
            InputImageDirectoryNotFoundException: 
                Raised when the image input directory is not found
            InputMetadataDirectoryNotFoundException:
                Raised when the metadata input directory is not found
        """
        
        try:
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_GCODE_DIRECTORY): 
                    raise InputGCodeDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_IMAGE_DIRECTORY): 
                    raise InputImageDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_METADATA_DIRECTORY):
                    raise InputMetadataDirectoryNotFoundException()
        except InputGCodeDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_GCODE_DIRECTORY)
        except InputImageDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd())
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_IMAGE_DIRECTORY)
        except InputMetadataDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd())
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_METADATA_DIRECTORY)
    
    @classmethod
    def _check_data(
            cls, 
            gcode_path: str, 
            gcode_name: str,
            image_path: str,
            image_name: str, 
            metadata_path: str, 
            metadata_name: str):
        """Method to check if the input files exists.

        Parameters:
            gcode_path (str): Path to the gcode file
            gcode_name (str): Name of the gcode file
            image_path (str): Path to the image
            image_name (str): Name of the image
            metadata_path (str): Path to the metadata file.
            metadata_name (str): Name of the metadata file.

        Raises:
            ExtractImageException: 
                Raised when the input image cannot be found
            ImageNotFileException: 
                Raised when the input image is not a file
            NonSupportedImageExtensionException:
                Raised when the input image has a non supported file extension
            ExtractGCodeFileException: 
                Raised when the input gcode file cannot be found
            GCodeNotFileException: 
                Raised when the input gcode is not a file
            NonSupportedGcodeExtensionException:
                Raised when the input gcode file has a non supported file 
                extension
            ExtractMetadataException:
                Raised when the input metadata file cannot be found
            MetadataNotFileException:
                Raised when the input metadata is not a file
            NonSupportedMetadataExtensionException:
                Raised when the input metadata file has a non supported file 
                extension
        """
        
        try:
            if os.path.exists(image_path):
                if os.path.isfile(image_path):
                    if not any(
                        [image_name.split(".")[1] == image_file_extension 
                         for image_file_extension in IMAGE_FILE_EXTENSIONS]):
                        raise NonSupportedImageExtensionException(
                            image_name.split(".")[0], 
                            image_name.split(".")[1], 
                            ", ".join(IMAGE_FILE_EXTENSIONS))
                else: 
                    raise ImageNotFileException(image_name)
            else: 
                raise ExtractImageException(image_name)
            
            if os.path.exists(gcode_path):
                if os.path.isfile(gcode_path):
                    if not gcode_name.split(".")[1] == GCODE_FILE_EXTENSION:
                        raise NonSupportedGcodeExtensionException(
                            gcode_name.split(".")[0], 
                            gcode_name.split(".")[1])
                else: 
                    raise GCodeNotFileException(gcode_name)
            else: 
                raise ExtractGCodeFileException(gcode_name)
            
            if not metadata_name == "":
                if os.path.exists(metadata_path):
                    if os.path.isfile(metadata_path):
                        if not (metadata_name.split(".")[1] 
                                == METADATA_FILE_EXTENSION):
                            raise NonSupportedMetadataExtensionException(
                                metadata_name.split(".")[0], 
                                metadata_name.split(".")[1])
                    else: 
                        raise MetadataNotFileException(metadata_name)
                else:
                    raise ExtractMetadataException(metadata_name)
            
        except (
            ExtractImageException, 
            ImageNotFileException, 
            NonSupportedImageExtensionException, 
            ExtractGCodeFileException, 
            GCodeNotFileException,  
            NonSupportedGcodeExtensionException, 
            ExtractMetadataException, 
            MetadataNotFileException, 
            NonSupportedMetadataExtensionException) as e:
            CommonPrints.system_out(e)