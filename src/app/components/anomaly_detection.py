import os
import sys
from typing import List

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.constants.constants import *
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
                train_neural_network: bool, 
                pretrained_model_name: str):
            Main algorithm to detect and classify 3d printing anomalies in 
            images.

    Raises:
        ModelOutputDirectoryNotFound: 
            Raised when the siamese neural network models input directory 
            is not found
        ModelNotFileException: 
            Raised when the user didn't specified the siamese neural 
            network to betrained and there is no model to be extracted
        ModelNotFoundException: 
            Raised when the user specified a model name which is not a 
            valid file
        InputGCodeDirectoryNotFoundException: 
            Raised when the gcode input directory is not found
        InputImageDirectoryNotFoundException: 
            Raised when the image input directory is not found
        InputMetadataDirectoryNotFoundException:
            Raised when the metadata input directory is not found
        TrainImagesDirectoryNotFoundException:
            Raised when the train images directory is not found
        TrainLabelsDirectoryNotFoundException:
            Raised when the train labels directory is not found
        TestImagesDirectoryNotFoundException:
            Raised when the test images directory is not found
        TestLabelsDirectoryNotFoundException:
            Raised when the test labels directory is not found
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
        TrainImagesNotFoundException: 
            Raised when there are no training images to be extracted
        TestImagesNotFoundException: 
            Raised when there are no testing images to be extracted
        TrainImageNotFileException: 
            Raised when any of the train images is not a file
        TestImageNotFileException: 
            Raised when any of the test images is not a file
        NonSupportedTrainImageExtensionException: 
            Raised when any of the train images has a non supported file 
            extension
        NonSupportedTestImageExtensionException: 
            Raised when any of the test images has a non supported file 
            extension
        NotEnumeratedTrainImagesException: 
            Raised when the training images are not correctly enumerated
        NotEnumeratedTestImagesException: 
            Raised when the testing images are not correctly enumerated
        TrainLabelsNotFoundException: 
            Raised when the training labels file doesn't exist
        TestLabelsNotFoundException: 
            Raised when the testing labels file doesn't exist
        TrainLabelsNotFileException: 
            Raised when the train labels is not a file
        TestLabelsNotFileException: 
            Raised when the test labels is not a file
        TrainLabelsZeroDataException: 
            Raised when there are no training labels to be extracted
        TestLabelsZeroDataException: 
            Raised when there are no testing labels to be extracted
        IncorrectTrainLabelsFormatException: 
            Raised when the training labels are not correctly formated
        IncorrectTestLabelsFormatException: 
            Raised when the testing labels are not correctly formated
        TrainLabelsIncorrectMatchTrainImagesException: 
            Raised when the training labels file contain more or less 
            labels than there are training images
        TestLabelsIncorrectMatchTestImagesException: _description_
            Raised when the testing labels file contain more or less 
            labels than there are testing images
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
            train_neural_network: bool, 
            pretrained_model_name: str) -> None:
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
            pretrained_model_name: Name of the pretrained neural network model
        """

        # Extract data
        (gcode_file, 
         image, 
         metadata_file, 
         trainX, 
         trainY, 
         testX, 
         testY, 
         model) = Extract.extract_process_data(
            gcode_name, 
            image_name, 
            metadata_name, 
            train_neural_network, 
            pretrained_model_name)
         
        if train_neural_network:
            (model, 
             history, 
             pair_train_len, 
             pair_test_len) = SiameseNeuralNetwork \
                 .construct_and_train_siamese_neural_network(
                     trainX, trainY, testX, testY)
        else:
            pair_train_len = None
            pair_test_len = None
            history = None

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
        
        # Defect classification
        (max_probability, max_probability_index) = SiameseNeuralNetwork \
            .predict(model, image, testX)
           
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
            # Classification
            trainX, 
            trainY, 
            testX, 
            testY, 
            pair_train_len, 
            pair_test_len, 
            MODEL_NAME if train_neural_network else pretrained_model_name, 
            history, 
            max_probability, 
            testX[max_probability_index], 
            # Extra data
            metadata_file)