class InputExecutionTypeNotSpecifiedException(Exception):
    """Raised when the user don't introduce the type of execution in the 
    initial command

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="You didn't introduced an execution type") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputExecutionTypeNotSpecifiedException: {}. "
            "Available executions: 1. --execution_type anomaly_detection"
        ).format(self.message)
    
class InputImageNameNotSpecifiedException(Exception):
    """Raised when the user don't introduce the image file name in the initial 
    command

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="You didn't introduced an image name") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputImageNameNotSpecifiedException: {}. "
            "It should be introduced with a prefix and the image name: "
            "--image_name image_name"
        ).format(self.message)
        
class InputGcodeNameNotSpecifiedException(Exception):
    """Raised when the user don't introduce the gcode file name in the initial 
    command

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="You didn't introduced a gcode name") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputGcodeNameNotSpecifiedException: {}. "
            "It should be introduced with a prefix and the gcode name: "
            "--gcode_name gcode_name"
        ).format(self.message)
        
class InputMetadataNameNotSpecifiedException(Exception):
    """Raised when the user don't introduce the metadata file name in the 
    initial command

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="You didn't introduced a metadata name") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputMetadataNameNotSpecifiedException: {}. "
            "No metadata will be introduced in the final report."
        ).format(self.message)
    
class InputReferenceObjectWidthNotSpecifiedException(Exception):
    """Raised when the user don't introduce the reference object with in the 
    initial command

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="You didn't introduced a reference object width") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputReferenceObjectWidthNotSpecifiedException: {}. "
            "The execution will use 18.74 as the default value which is the "
            "measure of a 2 cent coin in milimeters."
        ).format(self.message)