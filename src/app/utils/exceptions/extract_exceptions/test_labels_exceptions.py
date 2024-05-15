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
        return "TestLabelsNotFoundException: {}".format(self.message)
        
class TestLabelsNotFileException(Exception):
    """Raised when the test labels is not a file.

    Parameters:
        labels_file_name (str): Name of the test labels file
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            labels_file_name, 
            message=("The test labels is not a valid file. "
                     "Check the extension")) -> None:
        self.labels_file_name = labels_file_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TestLabelsNotFileException: {}. Test labels file name -> '{}'. "
        ).format(self.message, self.labels_file_name)
        
class NonSupportedTestLabelsExtensionException(Exception):
    """Raised when the test labels file has a non supported file extension.

    Parameters:
        labels_file_name (str): Name of the test labels file
        labels_file_name_extension (str): 
            The extension of the test labels file
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
            "NonSupportedTestLabelsExtensionException: The test labels file "
            "'{}' has a '{}' extension, which is not supported. This is the " 
            "test labels file supported extension: .txt"
        ).format(
            self.labels_file_name, 
            self.labels_file_name_extension)

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
        
class TestLabelsIncorrectFileNameException(Exception):
    """Raised when the testing labels have an incorrect file name.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="The testing labels have an incorrect file name"
            ) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TestLabelsIncorrectFileNameException: {}. The testing labels "
            "file should go by the name of 'testY'."
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