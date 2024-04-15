import os
import sys
import numpy as np
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.gcode_analizer import GCodeAnalizer
from app.components.low_contrast_detection import LowContrastDetection
from app.components.image_segmentation import ImageSegmetation
from app.components.image_generator import ImageGenerator
from app.components.error_detection import ErrorDetection

class AnomalyDetection(object):

    """Class containing the main algorithm to detect anomalies in 3d printed objects.

    Methods:
        anomaly_detection (gcode_name: str, image_name: str):
            Main algorithm to detect 3d printing anomalies in images.
    """

    @classmethod
    def anomaly_detection(
            cls, 
            gcode_name: str, 
            image_name: str, 
            reference_object_width: float):
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

        # Image segmentation
        segmented_image: np.ndarray = ImageSegmetation.segment_image(image)
        
        # Pixels per metric
        pixels_per_metric: float = ImageSegmetation.get_pixels_per_metric()
        
        # Middle coord of the 3d printed object in the original image
        middle_coords_3d_object = ImageSegmetation \
            .get_3d_object_middle_coords()
        
        print(middle_coords_3d_object)
                
        # Analize gcode file and extract data
        coords: List[List[object]] = GCodeAnalizer.extract_data(gcode_file)

        # Create perfect printed model based on gcode information
        perfect_model = ImageGenerator.generate_image(
            segmented_image, 
            middle_coords_3d_object, 
            coords, 
            pixels_per_metric, 
            reference_object_width)
        
        # Mask and error detection
        ErrorDetection.detect_errors(segmented_image, perfect_model)
        
        # Load results
