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
            "A metadata directory will be created at ../app/data/input/"
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
            "ExtractMetadataException: metadata file name -> '{}'. {}."
        ).format(self.metadata_name, self.message)
        
class MetadataNotFileException(Exception):
    """Raised when the input metadata is not a file.

    Parameters:
        metadata_name (str): The name of the metadata file given by the user
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
            "MetadataNotFileException: metadata file name -> '{}'. {}."
        ).format(self.metadata_name, self.message)
        
class NonSupportedMetadataExtensionException(Exception):
    """Raised when the input metadata file has a non supported file extension.

    Parameters:
        metadata_name (str): The name of the metadata file given by the user
        metadata_extension (str): 
            The extension of the metadata file given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            metadata_name, 
            metadata_extension, 
            message="") -> None:
        self.metadata_name = metadata_name
        self.metadata_extension = metadata_extension
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NonSupportedMetadataExtensionException: The metadata file '{}' " 
            "has a '{}' extension, which is not supported. This is the " 
            "metadata supported extension: .txt"
        ).format(
            self.metadata_name, 
            self.metadata_extension)