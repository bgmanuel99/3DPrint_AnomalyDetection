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
    """Class containing all the methods to create the final process report
    
    Methods:
        load_data (
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
                metadata_file: io.TextIOWrapper | str):
            Method to create the final process report
        _create_pdf_report (image_name: str, gcode_name: str):
            Method that inserts all the data into a report and saves it
        _insert_input_process_data  ():
            Method to insert the input process data
        _insert_impresion_defects_data ():
            Method to insert impresion defects detection data
        _insert_areas_calculations_data ():
            Method to insert areas calculations data
        _insert_text (
                x_coord: float = 0.0, 
                y_coord: float = 0.0, 
                units: float=cm, 
                font_type: str="Times-Roman", 
                font_size: str=11, 
                color: Color=colors.black, 
                alpha: float=1.0, 
                text_line: str=""):
            Method to transform a textobject along the way of creating the 
            final report
        _draw_text_and_reset_page (new_page: bool=False):
            Method to draw the text of the textobject to the final report and 
            if necessary create a new page for the report
        _load_images (
                images: tuple[tuple[str], tuple[np.ndarray]]):
            Method to load the process images to later read them and insert 
            them into the report
        _delete_loaded_images ():
            Method to delete the process images
        _check_output_directory ():
            Method to check if the output directory exists.

    Raises:
        OutputImageDirectoryNotFound:
            Raised when the image output directory is not found
    """
    
    _textobject: PDFTextObject = None
    _report: Canvas = None
    _report_width: float = None
    _report_height: float = None
    _image_name: str = None
    _gcode_name: str = None
    _metadata_name: str = None
    _reference_object_width: float = None
    _original_image: np.ndarray = None
    _ssim_max_score: float = None
    _pixels_per_metric: float = None
    _impresion_defects_total_diff: float = None
    _segmentation_defect_total_diff: float = None
    _infill_areas: List[List[object]] = None
    _ssim_max_score_reference_object: float = None
    _metadata_file: io.TextIOWrapper | str = None
    _image_paths: List[str] = None
    
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
        """Method to create the final process report

        Parameters:
            image_name (str): Original image name
            gcode_name (str): Gcode file name
            metadata_name (str): Metadata file name
            reference_object_width (float): Input reference object width
            original_image (np.ndarray): Original image
            perfect_model (np.ndarray): 
                Perfect model of the 3d printed object
            masked_3d_object (np.ndarray): 
                Image with the 3d printed object masked
            masked_3d_object_with_defects (np.ndarray): 
                Image with the 3d printed object masked and with its detected 
                defects
            ssim_max_score (float): 
                SSIM max score from the images of the 3d printed object and 
                its perfect models
            pixels_per_metric (float): 
                Value of pixels per metric used to create the the perfect 
                model with the max SSIM score
            impresion_defects_total_diff (float): 
                Total impresion error in the defect detection
            segmentation_defects_total_diff (float): 
                Total segmentation error in the defect detection
            infill_contours_image (np.ndarray): 
                Image with the enumerated internal areas of the 3d printed 
                object
            infill_areas (List[List[object]]): 
                Enumerated list with the calculated area of the internal areas 
                of the 3d printed object
            ssim_max_score_reference_object (float): 
                SSIM max score from the images of the reference object and its 
                perfect models
            metadata_file (io.TextIOWrapper | str): 
                Metadata file
        """
        
        cls._image_name = image_name
        cls._gcode_name = gcode_name
        cls._metadata_name = metadata_name
        cls._reference_object_width = reference_object_width
        cls._original_image = original_image
        cls._ssim_max_score = ssim_max_score
        cls._pixels_per_metric = pixels_per_metric
        cls._impresion_defects_total_diff = impresion_defects_total_diff
        cls._segmentation_defect_total_diff = segmentation_defects_total_diff
        cls._infill_areas = infill_areas
        cls._ssim_max_score_reference_object = ssim_max_score_reference_object
        cls._metadata_file = metadata_file
        
        # Check if output directory exists
        cls._check_output_directory()
        
        # Load the resultant images to the output folder
        cls._image_paths = cls._load_images((
            ("original_image", 
             "perfect_model", 
             "masked_3d_object", 
             "masked_3d_object_with_defects", 
             "infill_contours_image"), 
            # Flip the images vertically so when they are draw in the report 
            # they stay in the correct position as the report will draw them
            # upside down
            (cv2.flip(original_image, 0), 
             cv2.flip(perfect_model, 0), 
             cv2.flip(masked_3d_object, 0), 
             cv2.flip(masked_3d_object_with_defects, 0), 
             cv2.flip(infill_contours_image, 0))))
        
        try:
            # Create pdf report
            cls._create_pdf_report(image_name, gcode_name)
        except Exception as e:
            cls._delete_loaded_images()
            CommonPrints.system_out(e)
        finally:
            # Delete the resultant images from the output folder after inserting
            # them in the report pdf
            cls._delete_loaded_images()
        
    @classmethod
    def _create_pdf_report(cls, image_name: str, gcode_name: str) -> None:
        """Method that inserts all the data into a report and saves it

        Parameters:
            image_name (str): Original image name
            gcode_name (str): Gcode file name
        """
        
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
        cls._insert_input_process_data()
        cls._draw_text_and_reset_page(True)
        
        # Insert defects calculation data
        cls._insert_impresion_defects_data()
        cls._draw_text_and_reset_page(True)
        
        # Insert areas calculation data
        cls._insert_areas_calculations_data()
        cls._draw_text_and_reset_page(False)
        
        # Save report
        cls._report.save()
        
    @classmethod
    def _insert_input_process_data(cls) -> None:
        """Method to insert the input process data
        """
        
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
            cls._textobject.moveCursor(0.5*cm, 0.0)
            
            for line in cls._metadata_file.readlines():
                line = line.replace("\n", "")
                
                if cls._textobject.getY() >= cls._report_height - (2*cm):
                    cls._draw_text_and_reset_page(True)
                    cls._insert_text(
                        x_coord=3.5, y_coord=2.0, text_line=line)
                else:
                    cls._insert_text(y_coord=0.25, text_line=line)
        else:
            cls._insert_text(
                y_coord=0.25, 
                font_type="Times-Bold", 
                text_line="No input metadata was inserted")
            
    @classmethod
    def _insert_impresion_defects_data(cls) -> None:
        """Method to insert impresion defects detection data
        """
        
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
            text_line="{} %".format(cls._ssim_max_score))
        
        # Impresion error
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Impresion total error:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} %".format(cls._impresion_defects_total_diff))
        
        # Segmentation error
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Segmentation total error:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} %".format(cls._segmentation_defect_total_diff))
        
        # Original image
        cls._report.drawImage(
            image=cls._image_paths[0], 
            x=(cls._report_width-(5*cm))/2, 
            y=cls._report_height*0.33, 
            width=5*cm, 
            height=7*cm)
        cls._insert_text(
            x_coord=4.75, 
            y_coord=8, 
            text_line="Original image, size: {}".format(
                cls._original_image.shape))
            
        # Perfect model, masked 3d object, masked 3d object with errors
        cls._report.drawImage(
            image=cls._image_paths[1], 
            x=cls._report_width*0.1, 
            y=cls._report_height*0.63, 
            width=5*cm, 
            height=7*cm)
        cls._report.drawImage(
            image=cls._image_paths[2], 
            x=cls._report_width*0.39, 
            y=cls._report_height*0.63, 
            width=5*cm, 
            height=7*cm)
        cls._report.drawImage(
            image=cls._image_paths[3], 
            x=cls._report_width*0.68, 
            y=cls._report_height*0.63, 
            width=5*cm, 
            height=7*cm)
        cls._insert_text(
            x_coord=-5.5, 
            y_coord=8.8, 
            text_line=(
                "From left to right: [1] Perfect model, size: {}. " 
                "[2] Masked 3d printed object, size: {}."
            ).format(cls._original_image.shape, cls._original_image.shape))
        cls._insert_text(
            x_coord=3.9, 
            y_coord=0.25, 
            text_line="[3] Masked 3d printed object with defects, size {}." \
                .format(cls._original_image.shape))
        
    @classmethod
    def _insert_areas_calculations_data(cls) -> None:
        """Method to insert areas calculations data
        """
        
        # Impresion defects title
        cls._insert_text(
            2.0, 
            2.0, 
            font_size=16, 
            color=colors.red, 
            text_line="Internal areas")
        
        # SSIM max score
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.5, 
            font_type="Times-Bold", 
            text_line="· Structural similarity index measure max score:")
        cls._insert_text(
            x_coord=0.5, 
            y_coord=0.25,
            text_line="{} %".format(cls._ssim_max_score_reference_object))
        
        # Perfect model and internal areas
        cls._report.drawImage(
            image=cls._image_paths[4], 
            x=cls._report_width*0.27, 
            y=cls._report_height*0.18, 
            width=9.5*cm, 
            height=13.0*cm)
        
        # Internal areas calculations
        cls._insert_text(
            x_coord=-0.5, 
            y_coord=15.0, 
            font_type="Times-Bold", 
            text_line="· List of areas (mm2):")
        cls._textobject.moveCursor(0.5*cm, 0.0)
        init_list_coords = [
            cls._textobject.getX(), cls._textobject.getY()]
        offset = 2.5*cm
        for (i, area) in cls._infill_areas:
            if ((cls._textobject.getY() >= cls._report_height-(2*cm))
                and not (cls._textobject.getX() >= cls._report_width-(6*cm))):
                cls._textobject.setTextOrigin(
                    init_list_coords[0]+offset, 
                    cls._textobject.getY()
                    -(cls._textobject.getY()-init_list_coords[1]))
                init_list_coords = [
                    cls._textobject.getX(), cls._textobject.getY()]
                cls._insert_text(
                    y_coord=0.25,  
                    text_line="[{}] {}".format(i, area))
            elif ((cls._textobject.getY() >= cls._report_height-(2*cm)) 
                  and (cls._textobject.getX() >= cls._report_width-(6*cm))):
                cls._draw_text_and_reset_page(True)
                init_list_coords = (3*cm, 1.75*cm)
                cls._insert_text(
                    x_coord=3.0, 
                    y_coord=2.0,  
                    text_line="[{}] {}".format(i, area))
            else:
                cls._insert_text(
                    y_coord=0.25,  
                    text_line="[{}] {}".format(i, area))
    
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
        """Method to transform a textobject along the way of creating the 
        final report

        Parameters:
            x_coord (float, optional): 
                X coordinate to move the textobject. Defaults to 0.0.
            y_coord (float, optional): 
                Y coordinate to move the textobject. Defaults to 0.0.
            units (float, optional): 
                Type of unit to multiple to the coordinates in order to 
                transform them into points. Defaults to cm.
            font_type (str, optional): 
                Type of font of the text. Defaults to "Times-Roman".
            font_size (str, optional): 
                Size of the text. Defaults to 11.
            color (Color, optional): 
                Color of the text. Defaults to colors.black.
            alpha (float, optional): 
                Alpha value for the transparency of the text. Defaults to 1.0.
            text_line (str, optional): 
                New text line. Defaults to "".
        """
        
        cls._textobject.moveCursor(x_coord*units, y_coord*units)
        cls._textobject.setFont(font_type, font_size)
        cls._textobject.setFillColor(color)
        cls._textobject.setFillAlpha(alpha)
        cls._textobject.textLine(text_line)
        
    @classmethod
    def _draw_text_and_reset_page(cls, new_page: bool=False) -> None:
        """Method to draw the text of the textobject to the final report and 
        if necessary create a new page for the report

        Parameters:
            new_page (bool, optional): 
                Boolean value to acknowledge if a new page is needed. 
                Defaults to False.
        """
        
        cls._report.drawText(cls._textobject)
        if new_page:
            cls._report.showPage()
            cls._textobject = cls._report.beginText(0.0, 0.0)
    
    @classmethod
    def _load_images(
            cls, 
            images: tuple[tuple[str], tuple[np.ndarray]]) -> List[str]:
        """Method to load the process images to later read them and insert 
        them into the report

        Parameters:
            images (tuple[tuple[str], tuple[np.ndarray]]): 
                Tuple of tuples with the names of the images and the actual 
                images

        Returns:
            List[str]: A list with the paths where the images have been load to
        """
        
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
    def _delete_loaded_images(cls) -> None:
        """Method to delete the process images
        """
        
        for image_path in cls._image_paths:
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