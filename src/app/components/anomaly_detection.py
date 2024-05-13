import os
import sys
import cv2
import numpy as np
from typing import List
from keras.api.layers import Input, Lambda, Dense
from keras.api.models import Model
import imutils

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
                reference_object_width: float):
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
            reference_object_width: float) -> None:
        """Main algorithm to detect 3d printing anomalies in images.

        Parameters:
            gcode_name (str): Name of the input gcode file
            image_name (str): Name of the input image
            metadata_name (str): Metadata name for report
            reference_object_width (float): 
                Known real width of the reference object
        """
        
        train_images = []
        for i in range(0, 150):
            
            image = cv2.imread("{}{}{}.jpg".format(
                os.path.dirname(os.getcwd()), 
                "/data/classification/images/", 
                i))
            image = imutils.resize(image, width=120)
            train_images.append(image)
        print(train_images[0].shape)
        train_images = np.array(train_images)
        cv2.imshow("train image", train_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        labels = []
        labels_file = open("{}{}".format(
                os.path.dirname(os.getcwd()), 
                "/data/classification/labels/labels.txt"), "r")
        for line in labels_file.readlines():
            line = line.strip().replace("\n", "")
            labels.append(line)
        labels = np.array(labels)
        labels = labels.astype(np.uint8)
        
        train_images = train_images / 255.0
        
        print("[INFO] preparing positive and negative pairs...")
        (pairTrain, labelTrain) = SiameseNeuralNetwork.make_pairs(
            train_images, labels)
        print(len(pairTrain))
        
        # specify the shape of the inputs for our network
        IMG_SHAPE = (159, 120, 3)
        # specify the batch size and number of epochs
        BATCH_SIZE = 64
        EPOCHS = 5
        
        # configure the siamese network
        print("[INFO] building siamese network...")
        imgA = Input(shape=IMG_SHAPE)
        imgB = Input(shape=IMG_SHAPE)
        featureExtractor = SiameseNeuralNetwork._build_siamese_architecture(
            IMG_SHAPE)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        
        # finally, construct the siamese network
        distance = Lambda(SiameseNeuralNetwork._euclidean_distance, output_shape=(None, 1))([featsA, featsB])
        outputs = Dense(1, activation="sigmoid")(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        
        # compile the model
        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy"])
        model.summary()
        # train the model
        print("[INFO] training model...")
        history = model.fit(
            [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS)
        
        MODEL_PATH = "{}/data/classification/output/siamese_model.h5".format(os.path.dirname(os.getcwd()))
        PLOT_PATH = "{}/data/classification/output/siamese_model_plot.png".format(os.path.dirname(os.getcwd()))
        
        # serialize the model to disk
        print("[INFO] saving siamese model...")
        model.save(MODEL_PATH)
        # plot the training history
        print("[INFO] plotting training history...")
        SiameseNeuralNetwork._plot_training(history, PLOT_PATH)

        exit()

        # Extract data
        (gcode_file, 
         image, 
         metadata_file, 
         classification_images, 
         labels) = Extract.extract_process_data(
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