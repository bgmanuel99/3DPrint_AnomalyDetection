import os
import sys
import cv2
import numpy as np
from typing import List
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.utils.exceptions.load_exceptions import *
from app.common.common import CommonPrints
from app.utils.constants.constants import *

class Load(object):
    @classmethod
    def create_pdf_report(
            cls, 
            image_name: str, 
            gcode_name: str, 
            original_image: np.ndarray, 
            perfect_model: np.ndarray, 
            masked_3d_object: np.ndarray, 
            original_image_with_errors: np.ndarray, 
            metadata_name: np.ndarray) -> None:
        
        # Check if output directory exists
        cls._check_directory()
        
        # Load the resultant images to the output folder
        image_paths = cls._load_images((
            (
                "original_image", 
                "perfect_model", 
                "masked_3d_object", 
                "original_image_with_errors"), 
            (
                original_image, 
                perfect_model, 
                masked_3d_object, 
                original_image_with_errors)
        ))
        
        # Create pdf report
        canvas = Canvas(
            "{}{}{}_{}.{}".format(
                os.path.dirname(os.getcwd()), 
                output_directory_path, 
                image_name, 
                gcode_name, 
                output_report_extension), 
            pagesize=A4, 
            bottomup=0)
        
        width_A4, height_A4 = A4
        width_A4, height_A4 = width_A4/cm, height_A4/cm
        
        textobject = canvas.beginText(width_A4*0.5*cm, height_A4*0.1*cm)
        textobject.setFont("Times-Roman", 12)
        textobject.textLine("3D PRINTING DEFECT DETECTION REPORT")
        
        canvas.drawText(textobject)
        
        for image_path, offset in zip(image_paths, list(range(1, 5))):
            canvas.drawInlineImage(image_path, offset*cm, 1*cm, 1*cm, 1*cm)
          
        # for finishing a page and start drawing in a new one  
        # canvas.showPage()
            
        canvas.save()
        
        # Delete the resultant images from the output folder after inserting
        # them in the report pdf
        cls._delete_load_images(image_paths)
        
    @classmethod
    def _load_images(
            cls, 
            images: tuple[tuple[str], tuple[np.ndarray]]) -> List[str]:
        image_paths = []
        
        for (image_name, image) in zip(*images):
            image_path = "{}{}{}.{}".format(
                os.path.dirname(os.getcwd()), 
                output_directory_path, 
                image_name, 
                output_image_file_extension)
            
            cv2.imwrite(image_path, image)
            
            image_paths.append(image_path)
        
        return image_paths
    
    @classmethod
    def _delete_load_images(cls, image_paths: List[str]) -> None:
        for image_path in image_paths:
            if os.path.exists(image_path): 
                os.remove(image_path)
    
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
                + output_directory_path): 
                    raise OutputImageDirectoryNotFound()
        except OutputImageDirectoryNotFound as e:
            CommonPrints.system_out(e)