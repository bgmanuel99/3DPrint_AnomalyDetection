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
        
class TrainLabelsNotFoundException(Exception):
    """Raised when the training labels file doesn't exist.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "The training labels file doesn't exist")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsNotFoundException: {}. It should go by the name of "
            "trainY.txt"
        ).format(self.message)
        
class TrainLabelsNotFileException(Exception):
    """Raised when the train labels is not a file.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=("The train labels is not a valid file. "
                     "Check the extension")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsNotFileException: {}. It should go by the name of "
            "trainY.txt"
        ).format(self.message)
        
class TrainLabelsZeroDataException(Exception):
    """Raised when there are no training labels to be extracted.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "There are no labels to be extracted in the train labels "
                "file")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return "TrainLabelsZeroDataException: {}.".format(self.message)
        
class IncorrectTrainLabelsFormatException(Exception):
    """Raised when the training labels are not correctly formated.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The training labels are not correctly formated"
            ) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "IncorrectTrainLabelsFormatException: {}. The training labels "
            "should be introduced one for each line of the file."
        ).format(self.message)
        
class TrainLabelsIncorrectMatchTrainImagesException(Exception):
    """Raised when the training labels file contain more or less labels than 
    there are training images.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            number_of_labels, 
            number_of_images, 
            message="The training labels file don't match the number of images"
            ) -> None:
        self.number_of_labels = number_of_labels
        self.number_of_images = number_of_images
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsIncorrectMatchTrainImagesException: {}. "
            "Number of training labels -> {}, number of training images -> {}."
        ).format(self.message, self.number_of_labels, self.number_of_images)