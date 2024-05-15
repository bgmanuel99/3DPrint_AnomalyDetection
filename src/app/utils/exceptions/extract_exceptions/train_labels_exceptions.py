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
        return "TrainLabelsNotFoundException: {}".format(self.message)
        
class TrainLabelsNotFileException(Exception):
    """Raised when the train labels is not a file.

    Parameters:
        labels_file_name (str): Name of the train labels file
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            labels_file_name, 
            message=("The train labels is not a valid file. "
                     "Check the extension")) -> None:
        self.labels_file_name = labels_file_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsNotFileException: {}. Train labels file name -> '{}'. "
        ).format(self.message, self.labels_file_name)
        
class NonSupportedTrainLabelsExtensionException(Exception):
    """Raised when the train labels file has a non supported file extension.

    Parameters:
        labels_file_name (str): Name of the train labels file
        labels_file_name_extension (str): 
            The extension of the train labels file
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            labels_file_name, 
            labels_file_name_extension, 
            message="") -> None:
        self.labels_file_name = labels_file_name
        self.labels_file_name_extension = labels_file_name_extension
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NonSupportedTrainLabelsExtensionException: The train labels file "
            "'{}' has a '{}' extension, which is not supported. This is the " 
            "train labels file supported extension: .txt"
        ).format(
            self.labels_file_name, 
            self.labels_file_name_extension)
        
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
        
class TrainLabelsIncorrectFileNameException(Exception):
    """Raised when the training labels have an incorrect file name.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The training labels have an incorrect file name"
            ) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainLabelsIncorrectFileNameException: {}. The training labels "
            "file should go by the name of 'trainY'."
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