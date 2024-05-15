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
            "A gcode directory will be created at ../app/data/input"
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
        return (
            "ExtractGCodeFileException: gcode file name -> '{}'. {}."
        ).format(
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
        return "GCodeNotFileException: gcode file name -> '{}'. {}.".format(
            self.gcode_name, 
            self.message)
        
class NonSupportedGcodeExtensionException(Exception):
    """Raised when the input gcode file has a non supported file extension.

    Parameters:
        gcode_name (str): The name of the gcode file given by the user
        gcode_extension (str): 
            The extension of the gcode file given by the user
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            gcode_name, 
            gcode_extension, 
            message="") -> None:
        self.gcode_name = gcode_name
        self.gcode_extension = gcode_extension
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NonSupportedGcodeExtensionException: The gcode file '{}' has a " 
            "'{}' extension, which is not supported. This is the gcode " 
            "supported extension: .gcode"
        ).format(
            self.gcode_name, 
            self.gcode_extension)