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
        
class TestLabelsNotFoundException(Exception):
    """Raised when the testing labels file doesn't exist.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "The testing labels file doesn't exist")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TestLabelsNotFoundException: {}. It should go by the name of "
            "testY.txt"
        ).format(self.message)
        
class TestLabelsNotFileException(Exception):
    """Raised when the test labels is not a file.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=("The test labels is not a valid file. "
                     "Check the extension")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsNotFileException: {}. It should go by the name of "
            "testY.txt"
        ).format(self.message)
        
class TestLabelsZeroDataException(Exception):
    """Raised when there are no testing labels to be extracted.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "There are no labels to be extracted in the test labels "
                "file")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "TestLabelsZeroDataException: {}.".format(self.message)

class IncorrectTestLabelsFormatException(Exception):
    """Raised when the testing labels are not correctly formated.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The testing labels are not correctly formated"
            ) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "IncorrectTestLabelsFormatException: {}. The testing labels "
            "should be introduced one for each line of the file."
        ).format(self.message)
        
class TestLabelsIncorrectMatchTestImagesException(Exception):
    """Raised when the testing labels file contain more or less labels than 
    there are testing images.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            number_of_labels, 
            number_of_images, 
            message="The testing labels file don't match the number of images"
            ) -> None:
        self.number_of_labels = number_of_labels
        self.number_of_images = number_of_images
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TestLabelsIncorrectMatchTestImagesException: {}. "
            "Number of testing labels -> {}, number of testing images -> {}."
        ).format(self.message, self.number_of_labels, self.number_of_images)