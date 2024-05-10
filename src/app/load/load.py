import os
import io
import cv2
import numpy as np
from typing import List
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

from app.utils.exceptions.load_exceptions import *
from app.common.common_prints import CommonPrints
from app.utils.constants.constants import *

class Load(object):
    @classmethod
    def create_pdf_report(
            cls, 
            # Input process data
            image_name: str, 
            gcode_name: str, 
            metadata_name: str, 
            reference_object_width: str, 
            # Images
            original_image: np.ndarray, 
            perfect_model: np.ndarray, 
            masked_3d_object: np.ndarray, 
            masked_3d_object_with_defects: np.ndarray, 
            # Scores and errors
            ssim_max_score: float, 
            pixels_per_metric: float, 
            impresion_defects_total_diff: float, 
            segmentation_defects_total_diff: float, 
            # Images and data for areas
            infill_contours_image: np.ndarray, 
            infill_areas: List[List[object]], 
            ssim_max_score_reference_object: float, 
            # Extra data
            metadata_file: io.TextIOWrapper | str) -> None:
        
        # Check if output directory exists
        cls._check_directory()
        
        # Load the resultant images to the output folder
        image_paths = cls._load_images((
            (
                "original_image", 
                "perfect_model", 
                "masked_3d_object", 
                "masked_3d_object_with_defects"), 
            (
                original_image, 
                perfect_model, 
                masked_3d_object, 
                masked_3d_object_with_defects)
        ))
        
        # Create pdf report
        report = Canvas(
            "{}{}{}_{}.{}".format(
                os.path.dirname(os.getcwd()), 
                output_directory_path, 
                image_name.split(".")[0], 
                gcode_name.split(".")[0], 
                output_report_extension), 
            pagesize=A4, 
            bottomup=0)
        
        PAGE_WIDTH, PAGE_HEIGHT = A4
        
        report.setTitle("3D_printing_defect_detection_report")
        
        report_title = "3D PRINTING DEFECT DETECTION REPORT"
        report_title_width = stringWidth(report_title, "Times-Bold", 18)
        textobject = report.beginText(
            (PAGE_WIDTH-report_title_width)/2, PAGE_HEIGHT*0.1)
        textobject.setFont("Times-Bold", 18)
        textobject.textLine("3D PRINTING DEFECT DETECTION REPORT")
        
        report.drawText(textobject)
        
        # for image_path, offset in zip(image_paths, list(range(1, 5))):
        #     report.drawInlineImage(image_path, offset*cm, 1*cm, 1*cm, 1*cm)
          
        # for finishing a page and start drawing in a new one  
        # report.showPage()
            
        report.save()
        
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