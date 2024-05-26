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
        
        print("[INFO] Detecting defects")
        
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        CommonPrints.print_image(
            "segmented_3d_object", segmented_3d_object, 600, True)
        
        # Calculate ssim max score between the segmented 3d printed object and 
        # the perfect models
        ssim_max_score, ssim_max_score_index = CommonFunctionalities \
            .calculate_ssim_max_score(segmented_3d_object, perfect_models)
            
        # Apply subtract operation to the segmented 3d printed object and
        # the perfect model
        subtract = cv2.subtract(
            segmented_3d_object, perfect_models[ssim_max_score_index])
        
        CommonPrints.print_image("subtract", subtract, 600, True)
        
        colored_subtract = cv2.cvtColor(subtract, cv2.COLOR_GRAY2RGB)
        colored_subtract[
            np.all(colored_subtract==(255, 255, 255), axis=-1)
        ] = (251, 4, 131)
        
        CommonPrints.print_image(
            "colored_subtract", colored_subtract, 600, True)
        
        # Add original 3d object with the 3d impresion defects
        masked_3d_object_with_defects = cv2.addWeighted(
            masked_3d_object, 0.5, colored_subtract, 0.5, gamma=0)
        
        CommonPrints.print_image(
            "masked_3d_object_with_defects", 
            masked_3d_object_with_defects, 
            600, 
            True)
        
        return (
            masked_3d_object_with_defects, 
            ssim_max_score_index, 
            ssim_max_score)