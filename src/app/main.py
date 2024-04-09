import sys

from utils.constants import *
from components.anomaly_detection import AnomalyDetection

if __name__ == "__main__":
    match sys.argv[1]:
        case "anomaly_detection":
            AnomalyDetection.anomaly_detection(
                gcode_name=sys.argv[2], 
                image_name=sys.argv[3], 
                reference_object_width=float(sys.argv[4]))
            
"""
Si el usuario no introduce ningun dato de anchura para el objecto de referencia, entonces se pondra como default el ancho de la moneda de 2 centimos que seria el objeto de referencia que se da por entendido que deberia haber en la imagen
"""