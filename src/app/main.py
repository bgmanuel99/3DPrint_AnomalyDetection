import sys

from common.common_prints import CommonPrints
from utils.exceptions.main_exceptions import *
from components.anomaly_detection import AnomalyDetection

if __name__ == "__main__":
    input_app = ""
    execution_type = None
    image_name = None
    gcode_name = None
    metadata_name = None
    reference_object_width = None
    train_neural_network = None
    pretrained_model_name = None
    
    for arg in sys.argv:
        input_app = input_app + " " + arg
    
    input_transform = [
        (arg.strip().split(" ")) for arg in input_app.split("--")[1:]]
    
    try: 
        for input_prefix, input_name in input_transform:
            match input_prefix:
                case "execution_type":
                    execution_type = input_name
                case "image_name":
                    image_name = input_name
                case "gcode_name":
                    gcode_name = input_name
                case "metadata_name":
                    metadata_name = input_name
                case "reference_object_width":
                    reference_object_width = float(input_name)
                case "train_neural_network":
                    if input_name == "True":
                        train_neural_network = True
                    elif input_name == "False":
                        train_neural_network = False
                    else:
                        raise InputTrainNeuralNetworkInvalidDataException()
                case "pretrained_model_name":
                    pretrained_model_name = input_name
    except (
        ValueError, 
        InputTrainNeuralNetworkInvalidDataException) as e:
        CommonPrints.system_out(e)
    
    try:
        if not execution_type:
            raise InputExecutionTypeNotSpecifiedException()
        elif not image_name:
            raise InputImageNameNotSpecifiedException()
        elif not gcode_name:
            raise InputGcodeNameNotSpecifiedException()
        elif not metadata_name:
            metadata_name = ""
            raise InputMetadataNameNotSpecifiedException()
        elif not reference_object_width:
            reference_object_width = 18.74
            raise InputReferenceObjectWidthNotSpecifiedException()
        elif ((not train_neural_network 
              or train_neural_network == False) 
              and not pretrained_model_name):
            raise NeedOfNeuralNetworkModelException()
        elif train_neural_network == True and pretrained_model_name:
            raise TrainNeuralNetworkModelWithSpecifiedNameException()
        elif not train_neural_network:
            train_neural_network = False
            raise InputTrainNeuralNetworkNotSpecifiedException()
    except (
        InputExecutionTypeNotSpecifiedException, 
        InputImageNameNotSpecifiedException, 
        InputGcodeNameNotSpecifiedException, 
        NeedOfNeuralNetworkModelException, 
        TrainNeuralNetworkModelWithSpecifiedNameException) as e:
        CommonPrints.system_out(e)
    except (
        InputMetadataNameNotSpecifiedException, 
        InputReferenceObjectWidthNotSpecifiedException, 
        InputTrainNeuralNetworkNotSpecifiedException) as e:
        print(e)
                
    match execution_type:
        case "anomaly_detection":
            AnomalyDetection.anomaly_detection(
                gcode_name=gcode_name, 
                image_name=image_name, 
                metadata_name=metadata_name, 
                reference_object_width=reference_object_width, 
                train_neural_network=train_neural_network, 
                pretrained_model_name=pretrained_model_name)