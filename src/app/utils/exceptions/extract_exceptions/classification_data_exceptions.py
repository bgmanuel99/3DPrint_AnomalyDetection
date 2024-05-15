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

class TrainLabelsDirectoryNotFoundException(Exception):
    """Raised when the train labels directory is not found

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find train labels directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsDirectoryNotFoundException: {}. "
            "The directory for train labels will be created at ../app/data/ "
            "classification/"
        ).format(self.message)
        
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

class TestLabelsDirectoryNotFoundException(Exception):
    """Raised when the test labels directory is not found

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find test labels directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TestLabelsDirectoryNotFoundException: {}. "
            "The directory for test labels will be created at ../app/data/ "
            "classification/"
        ).format(self.message)