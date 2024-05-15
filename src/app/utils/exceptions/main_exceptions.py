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
        
class InputTrainNeuralNetworkNotSpecifiedException(Exception):
    """Raised when the user don't introduce the train neural network boolean 
    to acknowledge if the neural network should be train or not

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=("You didn't specified if the neural network should be "
                     "trained")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputTrainNeuralNetworkNotSpecifiedException: {}. "
            "The neural network won't be trained during the execution, "
            "using a pretrained neural network instead."
        ).format(self.message)
        
class InputTrainNeuralNetworkInvalidDataException(Exception):
    """Raised when the user introduce an invalid data type for the train 
    neural network prefix in the inital command

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "You specified a wrong type for the "
                "train_neural_network prefix")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "InputTrainNeuralNetworkInvalidDataException: {}. "
            "It should be either True or False"
        ).format(self.message)
        
class NeedOfNeuralNetworkModelException(Exception):
    """Raised when the user don't introduce the train neural network boolean 
    to acknowledge if the neural network should be train or not and don't 
    specify a pretrained neural network model name to be extracted.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "You didn't specified a neural network name.")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "NeedOfNeuralNetworkModelException: {}. Introduce the name of "
            "the model in the initial command as --neural_network_name name"
            ".h5 and the model itself to the ../app/data/classification/"
            "models/ directory"
        ).format(self.message)
        
class TrainNeuralNetworkModelWithSpecifiedNameException(Exception):
    """Raised when the user want's to train the network during the execution 
    but also inserted the name of a pretrained saved model

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message=(
                "You specified to the execution to train the siamese neural "
                "network while also passing a pretrained model name")) -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "TrainNeuralNetworkModelWithSpecifiedNameException: {}. Specify "
            "which of them you want the execution to do."
        ).format(self.message)