import sys

from common.common import CommonPrints
from utils.exceptions.main_exceptions import *
from components.anomaly_detection import AnomalyDetection

if __name__ == "__main__":
    input_app = ""
    execution_type = None
    image_name = None
    gcode_name = None
    metadata_name = None
    reference_object_width = None
    
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
    except ValueError as e:
        print(("InputPrefixException: You forgot to introduce a prefix or a"
               "value review your input command."))
        exit()
    
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
            raise InputReferenceObjectWidthNotSpecifiedException()
    except (
        InputExecutionTypeNotSpecifiedException, 
        InputImageNameNotSpecifiedException, 
        InputGcodeNameNotSpecifiedException) as e:
        CommonPrints.system_out(e)
    except (
        InputMetadataNameNotSpecifiedException, 
        InputReferenceObjectWidthNotSpecifiedException) as e:
        print(e)
    
    if not reference_object_width:
        reference_object_width = 18.74
                
    match execution_type:
        case "anomaly_detection":
            AnomalyDetection.anomaly_detection(
                gcode_name=gcode_name, 
                image_name=image_name, 
                metadata_name=metadata_name, 
                reference_object_width=reference_object_width)