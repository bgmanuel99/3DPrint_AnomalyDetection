import os
import sys
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.components.siamese_neural_network.siamese_neural_network import (
    SiameseNeuralNetwork)
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
                reference_object_width: float, 
                train_neural_network: bool):
            Main algorithm to detect 3d printing anomalies in images.

    Raises:
        InputGCodeDirectoryNotFoundException: 
            Raised when the gcode input directory is not found
        InputImageDirectoryNotFoundException: 
            Raised when the image input directory is not found
        InputMetadataDirectoryNotFoundException:
            Raised when the metadata input directory is not found
        ExtractImageException: 
            Raised when the input image cannot be found
        ImageNotFileException: 
            Raised when the input image is not a file
        NonSupportedImageExtensionException:
            Raised when the input image has a non supported file extension
        ExtractGCodeFileException: 
            Raised when the input gcode file cannot be found
        GCodeNotFileException: 
            Raised when the input gcode is not a file
        NonSupportedGcodeExtensionException:
            Raised when the input gcode file has a non supported file 
            extension
        ExtractMetadataException:
            Raised when the input metadata file cannot be found
        MetadataNotFileException:
            Raised when the input metadata is not a file
        NonSupportedMetadataExtensionException:
            Raised when the input metadata file has a non supported file 
            extension
        LowContrastDetectionException: 
            Raised when an image is low contrast and cannot be used in the 
            pipeline
        OutputImageDirectoryNotFound:
            Raised when the image output directory is not found
    """

    @classmethod
    def anomaly_detection(
            cls, 
            gcode_name: str, 
            image_name: str, 
            metadata_name: str, 
            reference_object_width: float, 
            train_neural_network: bool) -> None:
        """Main algorithm to detect 3d printing anomalies in images.

        Parameters:
            gcode_name (str): Name of the input gcode file
            image_name (str): Name of the input image
            metadata_name (str): Metadata name for report
            reference_object_width (float): 
                Known real width of the reference object
            train_neural_network: 
                Boolean to acknowledge whether to train or not the siamese 
                neural network
        """

        # Extract data
        (gcode_file, 
         image, 
         metadata_file) = Extract.extract_process_data(
            gcode_name, image_name, metadata_name)

        # Detect low contrast images
        LowContrastDetection.low_contrast_dectection(image)

        # Image segmentation
        (masked_3d_object, 
         ppm_degree_offset, 
         middle_coords_3d_object, 
         top_left_coord_3d_object, 
         reference_object_pixels_area, 
         ssim_max_score_reference_object) = ImageSegmetation.segment_image(
            image)
           
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
         ssim_max_score_3d_object, 
         impresion_defects_total_diff, 
         segmentation_defects_total_diff) = DefectsDetection.detect_defects(
            masked_3d_object, perfect_models)
         
        # Defect classification
         
        # Internal contours area calculation
        infill_contours_image, infill_areas = AreaCalculation.calculate_areas(
            image.shape, 
            perfect_models[ssim_max_score_index], 
            reference_object_width, 
            reference_object_pixels_area)
        
        # Load results
        Load.load_data(
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
            ssim_max_score_3d_object, 
            ppm_degree_offset[ssim_max_score_index], 
            impresion_defects_total_diff, 
            segmentation_defects_total_diff, 
            # Images and data for areas
            infill_contours_image, 
            infill_areas, 
            ssim_max_score_reference_object, 
            # Extra data
            metadata_file)