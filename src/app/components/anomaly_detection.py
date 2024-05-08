import os
import sys
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.extract.extract import Extract
from app.components.gcode_analizer import GCodeAnalizer
from app.components.image_generator import ImageGenerator
from app.components.area_calculation import AreaCalculation
from app.components.defects_detection import DefectsDetection
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
                metadata_name: str, 
                reference_object_width: float):
            Main algorithm to detect 3d printing anomalies in images.
    """

    @classmethod
    def anomaly_detection(
            cls, 
            gcode_name: str, 
            image_name: str, 
            metadata_name: str, 
            reference_object_width: float) -> None:
        """Main algorithm to detect 3d printing anomalies in images.

        Parameters:
            gcode_name (str): Name of the input gcode file
            image_name (str): Name of the input image
            metadata_name (str): Metadata name for report
            reference_object_width (float): 
                Known real width of the reference object
        """

        # Extract data
        gcode_file, image, metadata_file = Extract.extract_process_data(
            gcode_name, 
            image_name, 
            metadata_name)

        # Detect low contrast images
        LowContrastDetection.low_contrast_dectection(image)

        # Image segmentation
        (masked_3d_object, 
         ppm_degree_offset, 
         middle_coords_3d_object, 
         top_left_coord_3d_object, 
         reference_object_pixels_area) = ImageSegmetation.segment_image(image)
           
        # Analize gcode file and extract data
        coords: List[List[object]] = GCodeAnalizer.extract_data(gcode_file)

        # Create perfect printed models based on gcode information and pixels 
        # per metric values
        perfect_models = ImageGenerator.generate_images(
            image.shape[0:2], 
            middle_coords_3d_object, 
            top_left_coord_3d_object, 
            coords, 
            ppm_degree_offset, 
            reference_object_width)
        
        # Mask and defect detection
        (masked_3d_object_with_defects, 
         ssim_max_score_index, 
         ssim_max_score, 
         impresion_defects_total_diff, 
         segmentation_defects_total_diff) = DefectsDetection.detect_defects(
            masked_3d_object, perfect_models)
         
        # Internal contours area calculation
        infill_contours_image, infill_areas = AreaCalculation.calculate_areas(
            image.shape, 
            perfect_models[ssim_max_score_index], 
            reference_object_width, 
            reference_object_pixels_area)
        
        exit()
        
        # Load results
        Load.create_pdf_report(
            # Input process data
            image_name, 
            gcode_name, 
            metadata_name, 
            reference_object_width, 
            # Images
            image, 
            perfect_models[ssim_max_score_index], 
            masked_3d_object, 
            masked_3d_object_with_defects, 
            # Scores and errors
            ssim_max_score, 
            ppm_degree_offset[ssim_max_score_index], 
            impresion_defects_total_diff, 
            segmentation_defects_total_diff, 
            # Images and data for areas
            infill_contours_image, 
            infill_areas, 
            # Extra data
            metadata_file)