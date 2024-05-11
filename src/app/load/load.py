import os
import io
import cv2
import numpy as np
from typing import List
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.textobject import PDFTextObject

from app.utils.exceptions.load_exceptions import *
from app.common.common_prints import CommonPrints
from app.utils.constants.constants import *

class Load(object):
    
    _textobject: PDFTextObject = None
    _report: Canvas = None
    _report_width: float = None
    _report_height: float = None
    _image_name: str = None
    _gcode_name: str = None
    _metadata_name: str = None
    _reference_object_width: float = None
    _original_image: np.ndarray = None
    _perfect_model: np.ndarray = None
    _masked_3d_object: np.ndarray = None
    _masked_3d_object_with_defects: np.ndarray = None
    _ssim_max_score: float = None
    _pixels_per_metric: float = None
    _impresion_defects_total_diff: float = None
    _segmentation_defect_total_diff: float = None
    _infill_contours_image: np.ndarray = None
    _infill_areas: List[List[object]] = None
    _ssim_max_score_reference_object: float = None
    _metadata_file: io.TextIOWrapper | str = None
    
    @classmethod
    def load_data(
            cls, 
            # Input process data
            image_name: str, 
            gcode_name: str, 
            metadata_name: str, 
            reference_object_width: float, 
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
        cls._image_name = image_name
        cls._gcode_name = gcode_name
        cls._metadata_name = metadata_name
        cls._reference_object_width = reference_object_width
        cls._original_image = original_image
        cls._perfect_model = perfect_model
        cls._masked_3d_object = masked_3d_object
        cls._masked_3d_object_with_defects = masked_3d_object_with_defects
        cls._ssim_max_score = ssim_max_score
        cls._pixels_per_metric = pixels_per_metric
        cls._impresion_defects_total_diff = impresion_defects_total_diff
        cls._segmentation_defect_total_diff = segmentation_defects_total_diff
        cls._infill_contours_image = infill_contours_image
        cls._infill_areas = infill_areas
        cls._ssim_max_score_reference_object = ssim_max_score_reference_object
        cls._metadata_file = metadata_file
        
        # Check if output directory exists
        cls._check_output_directory()
        
        # Load the resultant images to the output folder
        image_paths = cls._load_images((
            ("original_image", 
             "perfect_model", 
             "masked_3d_object", 
             "masked_3d_object_with_defects"), 
            (original_image, 
             perfect_model, 
             masked_3d_object, 
             masked_3d_object_with_defects)))
        
        # Create pdf report
        cls._create_pdf_report(image_name, gcode_name)
        
        # Delete the resultant images from the output folder after inserting
        # them in the report pdf
        cls._delete_loaded_images(image_paths)
        
    @classmethod
    def _create_pdf_report(cls, image_name: str, gcode_name: str) -> None:
        cls._report = Canvas(
            "{}{}{}_{}.{}".format(
                os.path.dirname(os.getcwd()), 
                output_directory_path, 
                image_name.split(".")[0], 
                gcode_name.split(".")[0], 
                output_report_extension), 
            pagesize=A4, 
            bottomup=0)
        
        cls._report_width, cls._report_height = A4
        
        # Set file title
        cls._report.setTitle("3D printing defect detection report")
        
        # Set report title
        report_title = "3D PRINTING DEFECT DETECTION REPORT"
        report_title_width = stringWidth(report_title, "Times-Bold", 21)
        
        cls._textobject = cls._report.beginText(
            (cls._report_width-report_title_width)/2, cls._report_height*0.1)
        cls._insert_text(
            font_type="Times-Bold", 
            font_size=21, 
            text_line="3D PRINTING DEFECT DETECTION REPORT")
        
        # Reset textobject position to (0, 0) coordinates
        cls._textobject.setTextOrigin(0.0, 0.0)
        
        # Insert process input data
        cls._insert_process_input_data()
        cls._draw_text_and_reset_page()
        
        # Insert defects calculation data
        cls._insert_defects_calculation_data()
        cls._draw_text_and_reset_page()
        
        # for image_path, offset in zip(image_paths, list(range(1, 5))):
        #     report.drawInlineImage(image_path, offset*cm, 1*cm, 1*cm, 1*cm)
          
        # for finishing a page and start drawing in a new one  
        # report.showPage()
        
        # Save report
        cls._report.save()
        
    @classmethod
    def _insert_process_input_data(cls) -> None:
        # Input data title
        cls._insert_text(
            2.0, 5.0, font_size=16, color=colors.red, text_line="Input data")
        
        # Input image
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Input image:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="Name: {}".format(cls._image_name))
        cls._insert_text(
            y_coord=0.25,
            text_line="Size: {}".format(cls._original_image.shape))
        
        # Input gcode
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Input Gcode:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="Name: {}".format(cls._gcode_name))
        
        # Input reference object width
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Input reference object width:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25, 
            text_line="{} millimeters".format(cls._reference_object_width))
        
        # Input metadata
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Input metadata:")
        if cls._metadata_name != "":
            cls._insert_text(
                x_coord=0.5, 
                y_coord=0.25, 
                text_line="Name: {}".format(cls._metadata_name))
            cls._insert_text(y_coord=0.25, text_line="Data:")
            
            for line in cls._metadata_file.readlines():
                line = line.replace("\n", "")
                
                if cls._textobject.getY() >= cls._report_height - (2*cm):
                    cls._draw_text_and_reset_page()
                    cls._insert_text(
                        x_coord=3.0, y_coord=2.0, text_line=line)
                else:
                    cls._insert_text(y_coord=0.25, text_line=line)
        else:
            cls._insert_text(
                x_coord=0.5,
                y_coord=0.25, 
                font_type="Times-Bold", 
                text_line="No input metadata was inserted")
            
    @classmethod
    def _insert_defects_calculation_data(cls) -> None:
        # Impresion defects title
        cls._insert_text(
            2.0, 
            2.0, 
            font_size=16, 
            color=colors.red, 
            text_line="Impresion defects")
        
        # Pixels per metric
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Pixels per metric:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} pixels per {} millimeters".format(
                cls._pixels_per_metric, cls._reference_object_width))
        
        # SSIM max score
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Structural similarity index measure max score:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} %".format(
                cls._ssim_max_score))
        
        # Impresion error
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Impresion total error:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} %".format(
                cls._impresion_defects_total_diff))
        
        # Segmentation error
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Segmentation total error:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} %".format(
                cls._segmentation_defect_total_diff))
    
    @classmethod
    def _insert_text(
            cls, 
            x_coord: float = 0.0, 
            y_coord: float = 0.0, 
            units: float=cm, 
            font_type: str="Times-Roman", 
            font_size: str=11, 
            color: Color=colors.black, 
            alpha: float=1.0, 
            text_line: str="") -> None:
        cls._textobject.moveCursor(x_coord*units, y_coord*units)
        cls._textobject.setFont(font_type, font_size)
        cls._textobject.setFillColor(color)
        cls._textobject.setFillAlpha(alpha)
        cls._textobject.textLine(text_line)
        
    @classmethod
    def _draw_text_and_reset_page(cls) -> None:
        cls._report.drawText(cls._textobject)
        cls._report.showPage()
        cls._textobject = cls._report.beginText(0.0, 0.0)
    
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
    def _delete_loaded_images(cls, image_paths: List[str]) -> None:
        for image_path in image_paths:
            if os.path.exists(image_path): 
                os.remove(image_path)
    
    @classmethod
    def _check_output_directory(cls) -> None:
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