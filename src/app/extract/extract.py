import os
import sys
import cv2 as cv

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.exceptions import *

class Extract(object):
    
    @classmethod
    def extract_process_data(cls, file_name: str, image_name: str):
        # Check if directories exists
        cls._check_directories()
        
        # Check if data exists
        cls._check_data(file_name, image_name)
        
        # Extract data
        # TODO: use constants
        file = open("{}/data/input/gcode/{}.gcode".format(os.path.dirname(os.getcwd()), file_name), "r")
        
        image = cv.imread("{}/data/input/image/{}.png".format(os.path.dirname(os.getcwd()), image_name))
        
        return file, image
    
    @classmethod
    def _check_directories(cls):
        try:
            if not os.path.exists(os.path.dirname(os.getcwd()) + "/data/input/gcode"): raise GCodeDirectoryNotFoundException()
            if not os.path.exists(os.path.dirname(os.getcwd()) + "/data/input/image"): raise ImageDirectoryNotFoundException()
        except GCodeDirectoryNotFoundException as e:
            print(e)
        except ImageDirectoryNotFoundException as e:
            print(e)
    
    @classmethod
    def _check_data(cls, file_name: str, image_name: str):
        try:
            if os.path.exists(os.path.dirname(os.getcwd()) + "/data/input/gcode/{}.gcode".format(file_name)):
                if not os.path.isfile(os.path.dirname(os.getcwd()) + "/data/input/gcode/{}.gcode".format(file_name)):
                    raise GCodeNotFileException(file_name)
            else: raise ExtractGCodeFileException(file_name)
            
            if os.path.exists(os.path.dirname(os.getcwd()) + "/data/input/image/{}.png".format(image_name)):
                if not os.path.isfile(os.path.dirname(os.getcwd()) + "/data/input/image/{}.png".format(image_name)):
                    raise ImageNotFileException(image_name)
            else: raise ExtractImageException(image_name)
        except GCodeNotFileException as e:
            print(e)
        except ExtractGCodeFileException as e:
            print(e)
        except ImageNotFileException as e:
            print(e)
        except ExtractImageException as e:
            print(e)