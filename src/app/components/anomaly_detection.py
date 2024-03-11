import os
import sys
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.gcode_analizer.gcode_analizer import GCodeAnalizer
from app.components.low_contrast_detection.low_contrast_detection import (
    LowContrastDetection)
from app.components.generators.image_generator.turtle_image_generator import (
    TurtleImageGenerator)

class AnomalyDetection(object):

    """Class containing the main algorithm to detect anomalies in 3d printed objects.

    Methods:
        anomaly_detection (gcode_name: str, image_name: str):
            Main algorithm to detect 3d printing anomalies in images.
    """

    @classmethod
    def anomaly_detection(cls, gcode_name: str, image_name: str):
        """This is the main algorithm to detect 3d printing anomalies in images.

        Parameters:
            gcode_name (str): Name of the input gcode file
            image_name (str): Name of the input image
        """

        # Extract data
        gcode_file, image = Extract.extract_process_data(
            gcode_name, 
            image_name)

        # Detect low contrast images
        LowContrastDetection.low_contrast_dectection(image)

        # Analize gcode file and extract data
        coords: List[List[object]] = GCodeAnalizer.extract_data(gcode_file)

        # Create perfect printed model based on gcode information
        perfect_model = TurtleImageGenerator.generate_image(coords)
        
        # Image segmentation
        # Mask and error detection
        # Load results
