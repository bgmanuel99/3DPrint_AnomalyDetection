import cv2
import numpy as np
from typing import List

from app.common.common_functionalities import CommonFunctionalities
from app.common.common_prints import CommonPrints

class DefectsDetection(object):
    """Class containing methods to detect defects in the real 3d printed 
    object

    Methods:
        detect_defects (
                masked_3d_object: np.ndarray, 
                perfect_models: List[np.ndarray]):
            Method to detect exact defects between different perfect models of 
            the 3d impresion and a transformed and segmented image of the real 
            3d printed object
        _separate_impresion_and_segmentation_defects (
                masked_3d_object_shape: tuple[int], 
                cnts: tuple[np.ndarray]):
            Private method to separate in two images the impresion and 
            segmentation defects based on contour areas
        _calculate_impresion_and_segmentation_errors (
                impresion_defects: np.ndarray, 
                segmentation_defects: np.ndarray, 
                ssim_max_score: float):
            Private method to calculate the impresion and segmentation error 
            based on the SSIM max score
        _add_color_and_bounding_rect_to_defects (
                impresion_defects: np.ndarray, 
                segmentation_defects: np.ndarray, 
                cnts: tuple[np.ndarray]):
            Private method to add different colors to the defects and surround 
            them by there bounding rectangle
    """
    
    @classmethod
    def detect_defects(
            cls, 
            masked_3d_object: np.ndarray, 
            perfect_models: List[np.ndarray]) -> tuple[
                np.ndarray, int, float, float, float]:
        """Method to detect exact defects between different perfect models of 
        the 3d impresion and a transformed and segmented image of the real 3d 
        printed object

        Parameters:
            masked_3d_object (np.ndarray):
                Transformed and segmented image of the real 3d printed object
            perfect_models (List[np.ndarray]):
                Perfect models with sligth size differences of the 3d printed 
                object

        Returns:
            tuple[np.ndarray, int, float, float, float]:
                - A tuple with the image of the real 3d printed object with 
                the detected defectss
                - The SSIM max score index which will indicate the index in 
                the list of perfect models to know which of them was finally 
                used for the detection of the defects
                - The SSIM max score
                - The impresion defects error based on SSIM score
                - The segmentation defects error based on SSIM score
        """
        
        CommonPrints.print_image(
            "masked_3d_object", masked_3d_object, 600)
        
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        CommonPrints.print_image(
            "segmented_3d_object", segmented_3d_object, 600)
        
        # Calculate ssim max score between the segmented 3d printed object and 
        # the perfect models
        ssim_max_score, ssim_max_score_index = CommonFunctionalities \
            .calculate_ssim_max_score(segmented_3d_object, perfect_models)
        
        # Subtract the segmented 3d printed object to the perfect model
        # to extract impresion and segmentation defects
        subtract = cv2.subtract(
            perfect_models[ssim_max_score_index], segmented_3d_object)
        
        CommonPrints.print_image("subtract", subtract, 600)
        
        # Find and grab defects contours
        cnts = CommonFunctionalities.find_and_grab_contours(subtract)
        
        # Separate impresion and segmentation defects based on contour areas
        impresion_defects, segmentation_defects = cls \
            ._separate_impresion_and_segmentation_defects(
                masked_3d_object.shape, cnts)

        # Calculate defects percentage error based on ssim max score  
        impresion_defects_total_diff, segmentation_defects_total_diff = cls \
            ._calculate_impresion_and_segmentation_errors(
                impresion_defects, 
                segmentation_defects, 
                ssim_max_score)
        
        # Add color and bounding rectangles to the defects
        impresion_defects, segmentation_defects = cls \
            ._add_color_and_bounding_rect_to_defects(
                impresion_defects, segmentation_defects, cnts)
        
        CommonPrints.print_image(
            "Impresion defects", 
            impresion_defects, 
            600)
        CommonPrints.print_image(
           "Segmentation defects", 
           segmentation_defects, 
           600)
        
        # Add original 3d object with the 3d impresion defects
        masked_3d_object_with_defects = cv2.add(
            masked_3d_object, impresion_defects)
        
        CommonPrints.print_image(
            "masked_3d_object_with_defects", 
            masked_3d_object_with_defects, 
            600)
        
        return (
            masked_3d_object_with_defects, 
            ssim_max_score_index, 
            ssim_max_score, 
            impresion_defects_total_diff, 
            segmentation_defects_total_diff)
    
    @classmethod
    def _separate_impresion_and_segmentation_defects(
            cls, 
            masked_3d_object_shape: tuple[int], 
            cnts: tuple[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Method to separate in two images the impresion and segmentation 
        defects based on contour areas

        Parameters:
            masked_3d_object_shape (tuple[int]): 
                Shape of the image with the masked 3d object
            cnts (tuple[np.ndarray]): 
                Tuple of contours of the detected defects

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - Image with the impresion defects
                - Image with the segmentation defects
        """
        
        impresion_defects = np.zeros(
            masked_3d_object_shape, dtype=np.uint8)
        
        segmentation_defects = np.zeros(
            masked_3d_object_shape, dtype=np.uint8)
        
        for c in cnts:
            if cv2.contourArea(c) > 200:
                cv2.fillPoly(impresion_defects, [c], (255, 255, 255))
            else:
                cv2.fillPoly(
                    segmentation_defects, [c], (255, 255, 255))
                
        return impresion_defects, segmentation_defects
    
    @classmethod
    def _calculate_impresion_and_segmentation_errors(
            cls, 
            impresion_defects: np.ndarray, 
            segmentation_defects: np.ndarray, 
            ssim_max_score: float) -> tuple[float, float]:
        """Method to calculate the impresion and segmentation error based on 
        the SSIM max score

        Parameters:
            impresion_defects (np.ndarray): 
                Image with the impresion defects
            segmentation_defects (np.ndarray): 
                Image with the segmentation defects
            ssim_max_score (float): 
                SSIM max score

        Returns:
            tuple[float, float]: 
                - Impresion defects error
                - Segmentation defects error
        """
        
        impresion_defects_white_pixels = np.sum(
            impresion_defects == 255)
        
        segmentation_defects_white_pixels = np.sum(
            segmentation_defects == 255)
        
        total_ssim_diff = 1 - ssim_max_score
        
        total_white_pixels = (impresion_defects_white_pixels 
                              + segmentation_defects_white_pixels)
        
        impresion_defects_total_diff = (impresion_defects_white_pixels 
                                         * total_ssim_diff 
                                         / total_white_pixels)
        
        segmentation_defects_total_diff = (segmentation_defects_white_pixels 
                                            * total_ssim_diff 
                                            / total_white_pixels)
        
        return impresion_defects_total_diff, segmentation_defects_total_diff
    
    @classmethod
    def _add_color_and_bounding_rect_to_defects(
            cls, 
            impresion_defects: np.ndarray, 
            segmentation_defects: np.ndarray, 
            cnts: tuple[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Method to add different colors to the defects and surround them 
        by there bounding rectangle

        Parameters:
            impresion_defects (np.ndarray): 
                Image with the impresion defects
            segmentation_defects (np.ndarray): 
                Image with the segmentation defects
            cnts (tuple[np.ndarray]): 
                Tuple of contours of the detected defects

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - Image with the impresion defects colored and with a bounding 
                rectangle
                - Image with the segmentation defects colored and with a 
                bounding rectangle
        """
        
        # Change color of defects
        impresion_defects[
            np.all(impresion_defects == (255, 255, 255), axis=-1)
        ] = (251, 4, 131)
        
        segmentation_defects[
            np.all(segmentation_defects == (255, 255, 255), axis=-1)
        ] = (255, 0, 0)

        # Add defects bounding rectangles
        for c in cnts:
            if cv2.contourArea(c) > 200:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    impresion_defects, 
                    (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 
                    1)
            else:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    segmentation_defects, 
                    (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 
                    1)
                
        return impresion_defects, segmentation_defects