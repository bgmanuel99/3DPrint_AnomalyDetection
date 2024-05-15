class TrainImagesDirectoryNotFoundException(Exception):
    """Raised when the train images directory is not found

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find train images directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainImagesDirectoryNotFoundException: {}. "
            "The directory for train images will be created at ../app/data/ "
            "classification/"
        ).format(self.message)
        
class TrainImagesNotFoundException(Exception):
    """Raised when there are no training images to be extracted.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "There are no images to be extracted in the train images "
                "folder")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "TrainImagesNotFoundException: {}".format(self.message)
        
class TrainImageNotFileException(Exception):
    """Raised when any of the train images is not a file.

    Parameters:
        image_name (str): Image name
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            image_name, 
            message=("There are train images which are not valid files. "
                     "Check the extension")) -> None:
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "TrainImageNotFileException: {}. Image name -> '{}'.".format(
            self.message, self.image_name)
        
class NonSupportedTrainImageExtensionException(Exception):
    """Raised when any of the train images has a non supported file extension.

    Parameters:
        image_name (str): The name of the training image
        image_extension (str): The extension of the training image
        available_extension (List[str]): The list of available extensions
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            image_name, 
            image_extension, 
            available_extension, 
            message = "") -> None:
        self.image_name = image_name
        self.image_extension = image_extension
        self.available_extensions = available_extension
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NonSupportedTrainImageExtensionException: The image '{}' has a " 
            "'{}' extension, which is not supported. This are the image " 
            "supported extensions: {}"
        ).format(
            self.image_name, 
            self.image_extension, 
            self.available_extensions)
        
class NotEnumeratedTrainImagesException(Exception):
    """Raised when the training images are not correctly enumerated.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The training images are not correctly enumerated"
            ) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NotEnumeratedTrainImagesException: {}. The training images "
            "should have numeric names enumerated from 0 to N."
        ).format(self.message)