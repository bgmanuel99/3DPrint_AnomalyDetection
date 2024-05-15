class TestImagesDirectoryNotFoundException(Exception):
    """Raised when the test images directory is not found

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find test images directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TestImagesDirectoryNotFoundException: {}. "
            "The directory for test images will be created at ../app/data/ "
            "classification/"
        ).format(self.message)
        
class TestImagesNotFoundException(Exception):
    """Raised when there are no testing images to be extracted.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "There are no images to be extracted in the test images "
                "folder")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "TestImagesNotFoundException: {}".format(self.message)
        
class TestImageNotFileException(Exception):
    """Raised when any of the test images is not a file.

    Parameters:
        image_name (str): Image name
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            image_name, 
            message=("There are test images which are not valid files. "
                     "Check the extension")) -> None:
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "TestImageNotFileException: {}. Image name -> '{}'.".format(
            self.message, self.image_name)
        
class NonSupportedTestImageExtensionException(Exception):
    """Raised when any of the test images has a non supported file extension.

    Parameters:
        image_name (str): The name of the testing image
        image_extension (str): The extension of the testing image
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
            "NonSupportedTestImageExtensionException: The image '{}' has a " 
            "'{}' extension, which is not supported. This are the image " 
            "supported extensions: {}"
        ).format(
            self.image_name, 
            self.image_extension, 
            self.available_extensions)
        
class NotEnumeratedTestImagesException(Exception):
    """Raised when the testing images are not correctly enumerated.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The testing images are not correctly enumerated"
            ) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NotEnumeratedTestImagesException: {}. The testing images "
            "should be enumerated from 0 to N."
        ).format(self.message)