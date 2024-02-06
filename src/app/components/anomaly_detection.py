import os
import io
import sys
import numpy

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.low_contrast_detection.low_contrast_detection import LowContrastDetection
from app.components.gcode_analizer.gcode_analizer import GCodeAnalizer

class AnomalyDetection(object):
    
    @classmethod
    def anomaly_detection(cls, file_name: str, image_name: str):
        # Extract data
        file, image = Extract.extract_process_data(file_name, image_name)
        
        # Detect low contrast images
        LowContrastDetection.low_contrast_dectection(image)
        
        # Analize gcode file and extract data
        GCodeAnalizer.analize_gcode_file(file)
        
        # Create perfect model based on gcode
        
        
        # Image segmentation
        # Mask and error detection
        # Load results
        pass