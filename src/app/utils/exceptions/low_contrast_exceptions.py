class LowContrastDetectionException(Exception):
    """Raised when an image is low contrast and cannot be used in the pipeline.
    
    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The image has not enough contrast, please introduce a high contrast image to the pipeline"):
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "LowContrastDetectionException: {}".format(self.message)