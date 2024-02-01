class ExtractImageException(Exception):
    """Raised when the input image cannot be found

    Args:
        image_name  --> The name of the image given by the user
        message     --> Explanation message of the error
    """
    
    def __init__(self, image_name, message="The image could not be found or is in an incorrect folder"):
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    # TODO: __str__ function

class LowContrastDetectionException(Exception):
    """Raised when an image is low contrast and cannot be used in the pipeline
    
    Args:
        fraction_threshold  --> Threshold to determine whether the image has enough contrast
        message             --> Explanation message of the error
    """
    
    def __init__(self, fraction_threshold, message="The image has not enough contrast, please introduce a high contrast image to the pipeline"):
        self.fraction_threshold = fraction_threshold
        self.message = message
        
        super().__init__(self.message)
        
    # TODO: __str__ function