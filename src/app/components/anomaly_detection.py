import os
import sys

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.low_contrast_detection.low_contrast_detection import LowContrastDetection

class AnomalyDetection(object):
    
    @classmethod
    def anomaly_detection(cls, file_name: str, image_name: str):
        # Extract data
        Extract.extract_process_data(file_name, image_name)
        # Detect low contrast images
        # Create perfect model based on gcode
        # Image segmentation
        # Mask and error detection
        # Load results
        pass