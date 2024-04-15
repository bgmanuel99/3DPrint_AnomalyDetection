import sys

from utils.constants import *
from components.anomaly_detection import AnomalyDetection

if __name__ == "__main__":
    match sys.argv[1]:
        case "anomaly_detection":
            reference_object_width = (18.74 if len(sys.argv) == 4 else float 
                                      (sys.argv[4]))
            AnomalyDetection.anomaly_detection(
                gcode_name=sys.argv[2], 
                image_name=sys.argv[3], 
                reference_object_width=reference_object_width)