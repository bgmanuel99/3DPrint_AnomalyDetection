class InputImageDirectoryNotFoundException(Exception):
    """Raised when the image input directory is not found.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find images input directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputImageDirectoryNotFoundException: {}. "
            "An image directory will be created at ../app/data/input/"
        ).format(self.message)

class ExtractImageException(Exception):
    """Raised when the input image cannot be found.

    Parameters:
        image_name (str): The name of the image given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            image_name, 
            message=("The image could not be found or is in an incorrect "
                     "folder")) -> None:
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "ExtractImageException: image name -> '{}'. {}.".format(
            self.image_name, 
            self.message)
        
class ImageNotFileException(Exception):
    """Raised when the input image is not a file.

    Parameters:
        image_name (str): The name of the image given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            image_name, 
            message="The input image is not a valid file. Check the extension"
            ) -> None:
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "ImageNotFileException: image name -> '{}'. {}.".format(
            self.image_name, 
            self.message)
        
class NonSupportedImageExtensionException(Exception):
    """Raised when the input image has a non supported file extension.

    Parameters:
        image_name (str): The name of the image given by the user
        image_extension (str): The extension of the image given by the user
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
            "NonSupportedImageExtensionException: The image '{}' has a " 
            "'{}' extension, which is not supported. This are the image " 
            "supported extensions: {}"
        ).format(
            self.image_name, 
            self.image_extension, 
            self.available_extensions)