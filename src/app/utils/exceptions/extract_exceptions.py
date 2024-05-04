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
            "Check data folder, there should be a ../input/image/ directory."
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
        return "ExtractImageException: image name -> {}. {}.".format(
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
        return "ImageNotFileException: image name -> {}. {}.".format(
            self.image_name, 
            self.message)

class InputGCodeDirectoryNotFoundException(Exception):
    """Raised when the gcode input directory is not found.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find gcode input directory") -> None:
        self.message = message
        
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return (
            "InputGCodeDirectoryNotFoundException: {}. "
            "Check data folder, there should be a ../input/gcode/ directory."
        ).format(self.message)
    
class ExtractGCodeFileException(Exception):
    """Raised when the input gcode file cannot be found.

    Parameters:
        gcode_name (str): The name of the gcode file given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            gcode_name, 
            message=("The gcode file could not be found or is in an incorrect "
                     "folder")) -> None:
        self.gcode_name = gcode_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "ExtractGCodeFileException: gcode file name -> {}. {}.".format(
            self.gcode_name, 
            self.message)
    
class GCodeNotFileException(Exception):
    """Raised when the input gcode is not a file.

    Parameters:
        gcode_name (str): The name of the gcode file given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            gcode_name, 
            message="The input gcode is not a valid file. Check the extension"
            ) -> None:
        self.gcode_name = gcode_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "GCodeNotFileException: gcode file name -> {}. {}.".format(
            self.gcode_name, 
            self.message)
        
class InputMetadataDirectoryNotFoundException(Exception):
    """Raised when the metadata input directory is not found.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find metadata input directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputMetadataDirectoryNotFoundException: {}. "
            "Check data folder, there should be a ../input/metadata/ "
            "directory."
        ).format(self.message)
        
class ExtractMetadataException(Exception):
    """Raised when the input metadata file cannot be found.

    Parameters:
        gcode_name (str): The name of the metadata file given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            metadata_name, 
            message=("The metadata file could not be found or is in an "
                     "incorrect folder")) -> None:
        self.metadata_name = metadata_name
        self.message = message
        
        super().__init__(message)
        
    def __str__(self) -> str:
        return (
            "ExtractMetadataException: metadata file name -> {}. {}."
        ).format(self.metadata_name, self.message)
        
class MetadataNotFileException(Exception):
    """Raised when the input metadata is not a file.

    Parameters:
        gcode_name (str): The name of the metadata file given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            metadata_name, 
            message=("The input metadata file is not a valid file. "
                     "Check the extension")) -> None:
        self.metadata_name = metadata_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "MetadataNotFileException: metadata file name -> {}. {}."
        ).format(self.metadata_name, self.message)