import sys

from components.anomaly_detection import AnomalyDetection

if __name__ == "__main__":
    # TODO: pass image and gcode names or paths
    match sys.argv[1]:
        case "anomaly_detection":
            AnomalyDetection.anomaly_detection()
        case "gcode_generator":
            pass