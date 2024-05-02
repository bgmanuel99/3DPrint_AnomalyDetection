import os
import sys
import cv2
import numpy as np
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.load_exceptions import *
from app.common.common import CommonPrints
from app.utils.constants.constants import *

class Load(object):
    """Class for the process data load.

    Methods:
        create_pdf_report (image: numpy.ndarray): 
            Method to load process data.
        _check_directory:
            Private method to check if the output directory exists.
    """
    
    @classmethod
    def create_pdf_report(
            cls, 
            image_name: str, 
            gcode_name: str, 
            original_image: np.ndarray, 
            perfect_model: np.ndarray, 
            masked_3d_object: np.ndarray, 
            original_image_with_errors: np.ndarray, 
            metadata_path: np.ndarray) -> None:
        
        # Check if output directory exists
        cls._check_directory()
        
        # Create a new canvas for a pdf
        canvas = Canvas("Report.pdf", pagesize=A4)
        
    @classmethod
    def _load_images(
            cls, 
            original_image: np.ndarray, 
            perfect_model: np.ndarray, 
            masked_3d_object: np.ndarray, 
            original_image_with_errors: np.ndarray) -> tuple[str]:
        cv2.imwrite(
            "{}{}{}.{}".format(
                os.path.dirname(os.getcwd()), 
                output_image_directory_path, 
                "result", 
                output_image_file_extension), 
            original_image)
        
    # TODO: Añadir tipado a la devolucion del metodo
    @classmethod
    def _extract_images(cls):
        pass
    
    # TODO: Añadir check de borrado correcto de las imagenes con excepciones
    # pero que las excepciones no hagan salir del programa
    @classmethod
    def _delete_load_images(cls) -> None:
        pass
    
    @classmethod
    def _check_directory(cls) -> None:
        """Method to check if the output directory exists.

        Raises:
            OutputImageDirectoryNotFound: 
                Raised when the image output directory is not found
        """
        
        try:
            if not os.path.exists(
                os.path.dirname(os.getcwd()) 
                + output_image_directory_path): 
                    raise OutputImageDirectoryNotFound()
        except OutputImageDirectoryNotFound as e:
            CommonPrints.system_out(e)