class ModelOutputDirectoryNotFound(Exception):
    """Raised when the siamese neural network models input directory is not 
    found.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(self, message="Can't find models input directory") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "ModelOutputDirectoryNotFound: {}. "
            "An input directory will be created at ../app/data/classification/"
        ).format(self.message)                   

class ModelNotFoundException(Exception):
    """Raised when the user didn't specified the siamese neural network to be
    trained and there is no model to be extracted.

    Parameters:
        model_name (str): Name of the model to be extracted
        message (str): Explanation message of the error
    """
    
    def __init__(
        self, 
        model_name, 
        message="") -> None:
        self.model_name = model_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "ModelNotFoundException: The model you specified with name '{}' "
            "could not be found."
        ).format(self.model_name)
        
class ModelNotFileException(Exception):
    """Raised when the user specified a model name which is not a valid file.

    Parameters:
        model_name (str): Name of the model to be extracted
        message (str): Explanation message of the error
    """
    
    def __init__(
        self, 
        model_name, 
        message="") -> None:
        self.model_name = model_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "ModelNotFoundException: The model you specified with name '{}' "
            "is not a valid file."
        ).format(self.model_name)