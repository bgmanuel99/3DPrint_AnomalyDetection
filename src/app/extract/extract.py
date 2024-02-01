import os
import cv2 as cv

class Extract:
    
    @staticmethod
    def extract_process_data(file_name: str, image_name: str):
        image = cv.imread("{}/data/input/image/{}.png".format(os.path.dirname(os.getcwd().replace("\\", "/")), image_name))
        
        file = open("{}/data/input/gcode/{}.gcode".format(os.path.dirname(os.getcwd().replace("\\", "/")), file_name), "r")
        
        return file, image