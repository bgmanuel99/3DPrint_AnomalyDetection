import os
import sys

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.low_contrast_detection.low_contrast_detection import LowContrastDetection
from app.components.gcode_analizer.gcode_analizer import GCodeAnalizer

class AnomalyDetection(object):
    
    """Class containing the main algorithm to detect anomalies in 3d printed objects.
    
    Methods:
        anomaly_detection (file_name: str, image_name: str):
            Main algorithm to detect 3d printing anomalies in images.
    """
    
    @classmethod
    def anomaly_detection(cls, file_name: str, image_name: str):
        """This is the main algorithm to detect 3d printing anomalies in images.

        Parameters:
            file_name (str): Name of the input gcode file
            image_name (str): Name of the input image
        """
        
        # Extract data
        file, image = Extract.extract_process_data(file_name, image_name)
        
        # Detect low contrast images
        LowContrastDetection.low_contrast_dectection(image)
        
        # Analize gcode file and extract data
        vertices = GCodeAnalizer.analize_gcode_file(file)
        
        # Create perfect printed model based on gcode
        
        
        # Image segmentation
        # Mask and error detection
        # Load results
        pass