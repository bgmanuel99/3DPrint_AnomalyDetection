import sys

from components.anomaly_detection import AnomalyDetection

if __name__ == "__main__":
    # TODO: pass image and gcode names
    match sys.argv[1]:
        case "anomaly_detection":
            AnomalyDetection.anomaly_detection(file_name=sys.argv[2], image_name=sys.argv[3])
        case "gcode_generator":
            pass