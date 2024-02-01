import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from extract.extract import Extract
from components.low_contrast_detection.low_contrast_detection import LowContrastDetection

class AnomalyDetection:
    
    @staticmethod
    def anomaly_detection(file_name: str, image_name: str):
        # Extract data
        Extract.extract_process_data(file_name, image_name)
        # Detect low contrast images
        # Create perfect model based on gcode
        # Image segmentation
        # Mask and error detection
        # Load results
        pass