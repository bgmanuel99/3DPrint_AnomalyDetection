import os
import io
import cv2
import imutils
import numpy as np

from keras.api.models import Model, load_model
from app.utils.constants.constants import *
from app.common.common_prints import CommonPrints
from app.utils.exceptions.extract_exceptions.model_exceptions import *
from app.utils.exceptions.extract_exceptions.input_image_exceptions import *
from app.utils.exceptions.extract_exceptions.input_gcode_exceptions import *
from app.utils.exceptions.extract_exceptions.test_images_exceptions import *
from app.utils.exceptions.extract_exceptions.test_labels_exceptions import *
from app.utils.exceptions.extract_exceptions.train_images_exceptions import *
from app.utils.exceptions.extract_exceptions.train_labels_exceptions import *
from app.utils.exceptions.extract_exceptions.input_metadata_exceptions import *

class Extract(object):
    """Class for the process data extraction.
    
    Methods:
        extract_process_data (
                gcode_name: str, 
                image_name: str, 
                metadata_name: str, 
                train_neural_network: bool, 
                pretrained_model_name: str): 
            Method to extract process data.
        _check_models_directory ():
            Private method to check if the models input directory exists
        _check_models_data (
                model_path: str, 
                pretrained_model_name: str):
            Private method to check if the pretrained model exists and is a 
            file
        _check_directories (): 
            Private method to check if the directories of the input files
            exists.
        _check_input_data (
                gcode_path: str,
                gcode_name: str, 
                image_path: str,
                image_name: str, 
                metadata_path: str, 
                metadata_name: str): 
            Private method to check if the input files exists and are files.
        _check_classification_images_data (images_path: str):
            Private method to check if train or test classification images 
            data for the neural network exists and is correctly formated
        _check_classification_labels_data (labels_path: str):
            Private method to check if train or test classification labels 
            data for the neural network exists
        _extract_input_data (
                gcode_path: str, 
                image_path: str, 
                metadata_name: str, 
                metadata_path: str):
            Private method to extract input process data
        _extract_classification_images_data (
                images_path: str, 
                images_name_list: List[str]):
            Private method to extract the training or testing images data for 
            the siamese neural network
        _check_and_extract_classification_labels_data (
                number_of_images: int, 
                labels_path: str):
            Private method to extract and make last checks over the train and 
            testing classification labels data for the siamese neural network
        _extract_pretrained_model (model_path: str):
            Private method to extract the pretrained model
            
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
    """
    
    @classmethod
    def extract_process_data(
            cls, 
            gcode_name: str, 
            image_name: str, 
            metadata_name: str, 
            train_neural_network: bool, 
            pretrained_model_name: str) -> (
                tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    io.TextIOWrapper, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    None] 
                | tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    str, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    None]
                | tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    io.TextIOWrapper,
                    None, 
                    None, 
                    np.ndarray, 
                    None, 
                    Model]
                | tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    str,
                    None, 
                    None, 
                    np.ndarray, 
                    None, 
                    Model]):
        """Method to extract process data.

        Parameters:
            gcode_name (str): Name of the gcode file
            image_name (str): Name of the image
            metadata_name (str): Name of the metadata file
            train_neural_network (bool): 
                Boolean to acknowledge if the saimese neural network has to be 
                trained and to know if the data has to be extracted
            pretrained_model_name (str):
                Name of the pretrained neural network model

        Returns:
            (
                tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    io.TextIOWrapper, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    None] 
                | tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    str, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    np.ndarray, 
                    None]
                | tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    io.TextIOWrapper,
                    None, 
                    None, 
                    np.ndarray, 
                    None, 
                    Model]
                | tuple[
                    io.TextIOWrapper, 
                    np.ndarray, 
                    str,
                    None, 
                    None, 
                    np.ndarray, 
                    None, 
                    Model]
            ):
                Gcode file, image, metadata file, trainX, trainY, testX,  
                testY and model siamese neural network data. Metadata file and 
                nerual network data are optional to be extracted. If the model 
                is to trained during the execution this method will return the 
                training data if not it will return a pretrained model 
                specified by the user and the testX data for predictions
        """
        
        gcode_path: str = "{}{}{}{}".format(
            os.path.dirname(os.getcwd()), 
            INPUTS_FATHER_DIRECTORY_PATH, 
            INPUT_GCODE_DIRECTORY, 
            gcode_name)
        
        image_path: str = "{}{}{}{}".format(
            os.path.dirname(os.getcwd()), 
            INPUTS_FATHER_DIRECTORY_PATH, 
            INPUT_IMAGE_DIRECTORY, 
            image_name)
        
        metadata_path: str = "{}{}{}{}".format(
            os.path.dirname(os.getcwd()), 
            INPUTS_FATHER_DIRECTORY_PATH, 
            INPUT_METADATA_DIRECTORY, 
            metadata_name)
        
        if not train_neural_network:
            model_path: str = "{}{}{}".format(
                os.path.dirname(os.getcwd()), 
                MODEL_PATH, 
                pretrained_model_name)
            
            cls._check_models_directory()
            
            cls._check_models_data(model_path, pretrained_model_name)
            
            model = cls._extract_pretrained_model(model_path)
        
        if train_neural_network:
            trainX_path = (os.path.dirname(os.getcwd()) 
                           + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                           + TRAIN_IMAGES_DIRECTORY)
            
            trainY_path = (os.path.dirname(os.getcwd()) 
                           + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                           + TRAIN_LABELS_DIRECTORY
                           + TRAIN_LABELS_FILE_NAME)
            
            testY_path = (os.path.dirname(os.getcwd()) 
                            + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                            + TEST_LABELS_DIRECTORY
                            + TEST_LABELS_FILE_NAME)
        
        testX_path = (os.path.dirname(os.getcwd()) 
                      + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                      + TEST_IMAGES_DIRECTORY)
            
        # Check if directories exists
        cls._check_directories()
        
        # Check if data exists and are files
        cls._check_input_data(
            gcode_path, 
            gcode_name, 
            image_path, 
            image_name, 
            metadata_path, 
            metadata_name)
        
        if train_neural_network:
            # Check if train images data exists and have the right format
            train_images_list = cls._check_classification_images_data(
                trainX_path)
            
            # Check if train labels data exists and have the right format
            cls._check_classification_labels_data(trainY_path)
            
            # Check if test labels data exists and have the right format
            cls._check_classification_labels_data(testY_path)
            
        # Check if test images data exists and have the right format
        test_images_list = cls._check_classification_images_data(
            testX_path)
        
        # Extract input data
        (gcode_file, image, metadata_file) = cls._extract_input_data(
            gcode_path, image_path, metadata_name, metadata_path)
        
        testX = cls._extract_classification_images_data(
            testX_path, test_images_list)
        
        if train_neural_network:
            # Extract siamese neural network data
            trainX = cls._extract_classification_images_data(
                trainX_path, train_images_list)
            trainY = cls._check_and_extract_classification_labels_data(
                len(trainX), trainY_path)
            
            testY = cls._check_and_extract_classification_labels_data(
                len(testX),  testY_path)
        
        if train_neural_network:
            return (
                gcode_file, 
                image, 
                metadata_file, 
                trainX, 
                trainY, 
                testX, 
                testY, 
                None)
        else:
            return (
                gcode_file, 
                image, 
                metadata_file, 
                None, 
                None, 
                testX, 
                None, 
                model)
            
    @classmethod
    def _check_models_directory(cls) -> None:
        """Method to check if the models input directory exists

        Raises:
            ModelOutputDirectoryNotFound: 
                Raised when the siamese neural network models input directory 
                is not found
        """
        
        try:
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + MODEL_PATH):
                    raise ModelOutputDirectoryNotFound()
        except ModelOutputDirectoryNotFound as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + MODEL_PATH)
            
    @classmethod
    def _check_models_data(
            cls, 
            model_path: str, 
            pretrained_model_name: str) -> None:
        """Method to check if the pretrained model exists and is a file

        Parameters:
            model_path (str): Path to the pretrained model
            pretrained_model_name (str): Name of the pretrained model

        Raises:
            ModelNotFileException: 
                Raised when the user didn't specified the siamese neural 
                network to betrained and there is no model to be extracted
            ModelNotFoundException: 
                Raised when the user specified a model name which is not a 
                valid file
        """
        
        try:
            if os.path.exists(model_path):
                if not os.path.isfile(model_path):
                    raise ModelNotFileException(pretrained_model_name)
            else:
                raise ModelNotFoundException(pretrained_model_name)
        except (
            ModelNotFileException, 
            ModelNotFoundException) as e:
            CommonPrints.system_out(e)
            
    @classmethod
    def _check_directories(cls):
        """Method to check if the directories of the input files and 
        train data exists.

        Raises:
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
        """
        
        try:
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_GCODE_DIRECTORY): 
                    raise InputGCodeDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_IMAGE_DIRECTORY): 
                    raise InputImageDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_METADATA_DIRECTORY):
                    raise InputMetadataDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TRAIN_IMAGES_DIRECTORY):
                    raise TrainImagesDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TRAIN_LABELS_DIRECTORY):
                    raise TrainLabelsDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TEST_IMAGES_DIRECTORY):
                    raise TestImagesDirectoryNotFoundException()
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TEST_LABELS_DIRECTORY):
                    raise TestLabelsDirectoryNotFoundException()
        except InputGCodeDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_GCODE_DIRECTORY)
        except InputImageDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd())
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_IMAGE_DIRECTORY)
        except InputMetadataDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd())
                + INPUTS_FATHER_DIRECTORY_PATH
                + INPUT_METADATA_DIRECTORY)
        except TrainImagesDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TRAIN_IMAGES_DIRECTORY)
        except TrainLabelsDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TRAIN_LABELS_DIRECTORY)
        except TestImagesDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TEST_IMAGES_DIRECTORY)
        except TestLabelsDirectoryNotFoundException as e:
            print(e)
            os.mkdir(
                os.path.dirname(os.getcwd()) 
                + FATHER_CLASSIFICATION_DIRECTORY_PATH 
                + TEST_LABELS_DIRECTORY)
    
    @classmethod
    def _check_input_data(
            cls, 
            gcode_path: str, 
            gcode_name: str,
            image_path: str,
            image_name: str, 
            metadata_path: str, 
            metadata_name: str):
        """Method to check if the input files exists.

        Parameters:
            gcode_path (str): Path to the gcode file
            gcode_name (str): Name of the gcode file
            image_path (str): Path to the image
            image_name (str): Name of the image
            metadata_path (str): Path to the metadata file.
            metadata_name (str): Name of the metadata file.

        Raises:
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
        """
        
        try:
            if os.path.exists(image_path):
                if os.path.isfile(image_path):
                    if not any(
                        [image_name.split(".")[1] == image_file_extension 
                         for image_file_extension in IMAGE_FILE_EXTENSIONS]):
                        raise NonSupportedImageExtensionException(
                            image_name.split(".")[0], 
                            image_name.split(".")[1], 
                            ", ".join(IMAGE_FILE_EXTENSIONS))
                else: 
                    raise ImageNotFileException(image_name)
            else: 
                raise ExtractImageException(image_name)
            
            if os.path.exists(gcode_path):
                if os.path.isfile(gcode_path):
                    if not gcode_name.split(".")[1] == GCODE_FILE_EXTENSION:
                        raise NonSupportedGcodeExtensionException(
                            gcode_name.split(".")[0], 
                            gcode_name.split(".")[1])
                else: 
                    raise GCodeNotFileException(gcode_name)
            else: 
                raise ExtractGCodeFileException(gcode_name)
            
            if not metadata_name == "":
                if os.path.exists(metadata_path):
                    if os.path.isfile(metadata_path):
                        if not (metadata_name.split(".")[1] 
                                == METADATA_FILE_EXTENSION):
                            raise NonSupportedMetadataExtensionException(
                                metadata_name.split(".")[0], 
                                metadata_name.split(".")[1])
                    else: 
                        raise MetadataNotFileException(metadata_name)
                else:
                    raise ExtractMetadataException(metadata_name)
            
        except (
            ExtractImageException, 
            ImageNotFileException, 
            NonSupportedImageExtensionException, 
            ExtractGCodeFileException, 
            GCodeNotFileException,  
            NonSupportedGcodeExtensionException, 
            ExtractMetadataException, 
            MetadataNotFileException, 
            NonSupportedMetadataExtensionException) as e:
            CommonPrints.system_out(e)
        
    @classmethod
    def _check_classification_images_data(cls, images_path: str) -> List[str]:
        """Method to check if train or test classification images data for the 
        neural network exists and is correctly formated

        Parameters:
            images_path (str): Path to the train or testing images

        Raises:
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

        Returns:
            List[str]: List with the train or test sorted images names
        """
        
        try:
            if "trainX" in images_path:
                typeOfDirectory = "train"
            elif "testX" in images_path:
                typeOfDirectory = "test"
                
            images_list = os.listdir(images_path)
            
            if len(images_list) == 0:
                if typeOfDirectory == "train":
                    raise TrainImagesNotFoundException()
                elif typeOfDirectory == "test":
                    raise TestImagesNotFoundException()
            
            for image_name in images_list:
                if not os.path.isfile(images_path + image_name):
                    if typeOfDirectory == "train":
                        raise TrainImageNotFileException(image_name)
                    elif typeOfDirectory == "test":
                        raise TestImageNotFileException(image_name)
                
                if not any(
                    [image_name.split(".")[1] == image_file_extension 
                    for image_file_extension in IMAGE_FILE_EXTENSIONS]):
                    if typeOfDirectory == "train":
                        raise NonSupportedTrainImageExtensionException(
                            image_name.split(".")[0], 
                            image_name.split(".")[1], 
                            ", ".join(IMAGE_FILE_EXTENSIONS))
                    elif typeOfDirectory == "test":
                        raise NonSupportedTestImageExtensionException(
                            image_name.split(".")[0], 
                            image_name.split(".")[1], 
                            ", ".join(IMAGE_FILE_EXTENSIONS))

                if not image_name.split(".")[0].isdigit():
                    if typeOfDirectory == "train":
                        raise NotEnumeratedTrainImagesException()
                    elif typeOfDirectory == "test":
                        raise NotEnumeratedTestImagesException()
            
            images_list = sorted([
                [int(image_name.split(".")[0]), image_name.split(".")[1]] 
                for image_name in images_list])
            
            images_list = [
                "{}.{}".format(image_name, extension) 
                for (image_name, extension) in images_list]
            
            return images_list
        except (
            TrainImagesNotFoundException, 
            TestImagesNotFoundException, 
            TrainImageNotFileException, 
            TestImageNotFileException, 
            NonSupportedTrainImageExtensionException, 
            NonSupportedTestImageExtensionException, 
            NotEnumeratedTrainImagesException, 
            NotEnumeratedTestImagesException) as e:
            CommonPrints.system_out(e)
    
    @classmethod
    def _check_classification_labels_data(cls, labels_path: str):
        """Method to check if train or test classification labels data for the 
        neural network exists

        Parameters:
            labels_path (str): Path to the train or testing labels file

        Raises:
            TrainLabelsNotFoundException: 
                Raised when the training labels file doesn't exist
            TestLabelsNotFoundException: 
                Raised when the testing labels file doesn't exist
            TrainLabelsNotFileException: 
                Raised when the train labels is not a file
            TestLabelsNotFileException: 
                Raised when the test labels is not a file
        """
        
        try:
            if "trainY.txt" in labels_path:
                typeOfDirectory = "train"
            elif "testY.txt" in labels_path:
                typeOfDirectory = "test"
            
            if not os.path.exists(labels_path):
                if typeOfDirectory == "train":
                    raise TrainLabelsNotFoundException()
                elif typeOfDirectory == "test":
                    raise TestLabelsNotFoundException()
                
            if not os.path.isfile(labels_path):
                if typeOfDirectory == "train":
                    raise TrainLabelsNotFileException()
                elif typeOfDirectory == "test":
                    raise TestLabelsNotFileException()
        except (
            TrainLabelsNotFoundException, 
            TestLabelsNotFoundException, 
            TrainLabelsNotFileException, 
            TestLabelsNotFileException) as e:
            CommonPrints.system_out(e)
        
    @classmethod
    def _extract_input_data(
            cls, 
            gcode_path: str, 
            image_path: str, 
            metadata_name: str, 
            metadata_path: str) -> (
                tuple[io.TextIOWrapper, np.ndarray, io.TextIOWrapper] 
                | tuple[io.TextIOWrapper, np.ndarray, str]):
        """Method to extract input process data

        Parameters:
            gcode_name (str): Name of the gcode file
            image_name (str): Name of the image
            metadata_name (str): Name of the metadata file
            metadata_path (str): Metadata file path

        Returns:
            (
                tuple[io.TextIOWrapper, np.ndarray, io.TextIOWrapper] 
                | tuple[io.TextIOWrapper, np.ndarray, str]
            ): 
                Gcode file, original image and metadata file. Metadata file is 
                optional.
        """
        
        gcode_file = open(gcode_path, "r")
        
        image = cv2.imread(image_path)
        
        if not metadata_name == "":
            metadata_file = open(metadata_path, "r")
            
            return gcode_file, image, metadata_file
        else:
            return gcode_file, image, metadata_name
        
    @classmethod
    def _extract_classification_images_data(
            cls, 
            images_path: str, 
            images_name_list: List[str]) -> np.ndarray:
        """Method to extract the training or testing images data for the 
        siamese neural network

        Parameters:
            images_path (str): Path to the train or testing images folder
            images_name_list (List[str]): 
                Sorted list with the train or testing images names

        Returns:
            np.ndarray: A numpy array containing the training or testing images
        """
        
        images = []
        for image_name in images_name_list:
            image = cv2.imread(
                "{}{}".format(images_path, image_name))
            image = imutils.resize(image, width=120)
            images.append(image)
        images = np.array(images)
        
        return images
        
    @classmethod
    def _check_and_extract_classification_labels_data(
            cls, 
            number_of_images: int, 
            labels_path: str) -> np.ndarray:
        """Method to extract and make last checks over the train and testing 
        classification labels data for the siamese neural network

        Parameters:
            number_of_images (int): Number of train or testing images extracted
            labels_path (str): Path to the train or testing labels file

        Raises:
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
        Returns:
            np.ndarray: A numpy array containing the train or testing labels
        """
        
        try:
            if "trainY.txt" in labels_path:
                typeOfDirectory = "train"
            elif "testY.txt" in labels_path:
                typeOfDirectory = "test"
                    
            labels = []
            labels_file = open("{}".format(labels_path), "r")
            for label in labels_file.readlines():
                label = label.strip().replace("\n", "")
                labels.append(label)
                
            if len(labels) == 0:
                if typeOfDirectory == "train":
                    raise TrainLabelsZeroDataException()
                elif typeOfDirectory == "test":
                    raise TestLabelsZeroDataException()
                
            for label in labels:
                if len(label) > 1:
                    if typeOfDirectory == "train":
                        raise IncorrectTrainLabelsFormatException()
                    elif typeOfDirectory == "test":
                        raise IncorrectTestLabelsFormatException()
                    
            if len(labels) != number_of_images:
                if typeOfDirectory == "train":
                    raise TrainLabelsIncorrectMatchTrainImagesException(
                        len(labels), number_of_images)
                elif typeOfDirectory == "test":
                    raise TestLabelsIncorrectMatchTestImagesException(
                        len(labels), number_of_images)
            
            labels = np.array(labels)
            labels = labels.astype(np.uint8)
            
            return labels
        except (
            TrainLabelsZeroDataException, 
            TestLabelsZeroDataException, 
            IncorrectTrainLabelsFormatException, 
            IncorrectTestLabelsFormatException, 
            TrainLabelsIncorrectMatchTrainImagesException, 
            TestLabelsIncorrectMatchTestImagesException) as e:
            CommonPrints.system_out(e)
            
    @classmethod
    def _extract_pretrained_model(cls, model_path: str) -> Model:
        """Method to extract the pretrained model

        Parameters:
            model_path (str): Path to the pretrained model

        Returns:
            Model: Pretrained model
        """
        
        return load_model(model_path)