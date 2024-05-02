import os
import sys
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.gcode_analizer import GCodeAnalizer
from app.components.image_generator import ImageGenerator
from app.components.error_detection import ErrorDetection
from app.components.image_segmentation import ImageSegmetation
from app.components.low_contrast_detection import LowContrastDetection
from app.load.load import Load

class AnomalyDetection(object):

    """Class containing the main algorithm to detect anomalies in 3d printed 
    objects.

    Methods:
        anomaly_detection (
                gcode_name: str, 
                image_name: str, 
                metadata_path: str, 
                reference_object_width: float):
            Main algorithm to detect 3d printing anomalies in images.
    """

    @classmethod
    def anomaly_detection(
            cls, 
            gcode_name: str, 
            image_name: str, 
            metadata_path: str, 
            reference_object_width: float) -> None:
        """Main algorithm to detect 3d printing anomalies in images.

        Parameters:
            gcode_name (str): Name of the input gcode file
            image_name (str): Name of the input image
            metadata_path (str): Metadata path for report
            reference_object_width (float): 
                Known real world width of the reference object
        """

        # Extract data
        gcode_file, image = Extract.extract_process_data(
            gcode_name, 
            image_name)

        # Detect low contrast images
        LowContrastDetection.low_contrast_dectection(image)

        # Image segmentation
        (masked_3d_object, 
         ppm_degree_offset, 
         middle_coords_3d_object, 
         top_left_coord_3d_object) = ImageSegmetation.segment_image(image)
                
        # Analize gcode file and extract data
        coords: List[List[object]] = GCodeAnalizer.extract_data(gcode_file)

        # Create perfect printed models based on gcode information
        perfect_models = ImageGenerator.generate_images(
            image.shape[0:2], 
            middle_coords_3d_object, 
            top_left_coord_3d_object, 
            coords, 
            ppm_degree_offset, 
            reference_object_width)
        
        # Mask and error detection
        original_image_with_errors, ssim_max_score_index = ErrorDetection \
            .detect_errors(masked_3d_object, perfect_models, ppm_degree_offset)
        
        # Load results
        Load.create_pdf_report(
            image_name, 
            gcode_name, 
            image, 
            perfect_models[ssim_max_score_index], 
            masked_3d_object, 
            original_image_with_errors, 
            metadata_path)