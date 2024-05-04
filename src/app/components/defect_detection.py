import os
import sys
import cv2
import numpy as np
from typing import List
from skimage.metrics import structural_similarity

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.common.common import CommonPrints, CommonFunctionalities

class DefectDetection(object):
    """Class containing the method to detect defects in the real 3d printed 
    object

    Methods:
        detect_defects (
                cls, 
                masked_3d_object: np.ndarray, 
                perfect_models: List[np.ndarray], 
                ppm_degree_offset: List[float]):
            Method to detect exact defects between different perfect models of 
            the 3d impresion and a transformed and segmented image of the real 
            3d printed object
    """
    
    @classmethod
    def detect_defects(
            cls, 
            masked_3d_object: np.ndarray, 
            perfect_models: List[np.ndarray], 
            ppm_degree_offset: List[float]) -> tuple[np.ndarray, int, float]:
        """Method to detect exact defects between different perfect models of 
        the 3d impresion and a transformed and segmented image of the real 3d 
        printed object

        Parameters:
            masked_3d_object (np.ndarray):
                Transformed and segmented image of the real 3d printed object
            perfect_models (List[np.ndarray]):
                Perfect models with sligth size differences of the 3d printed 
                object
            ppm_degree_offset (List[float]):
                A list of the pixels per metric values containing the real 
                extracted value and others with a slight increase and decrease 
                value representing the degree error when the image of the real
                3d printed object is taken

        Returns:
            tuple[np.ndarray, int, float]:
                - A tuple with the image of the real 3d printed object with 
                the detected defectss
                - The max SSIM score index which will indicate the index in 
                the list of perfect models to know which of them was finally 
                used for the detection of the defects
                - The max SSIM score
        """
        
        ssim_max_score = 0
        ssim_max_score_index = 0
        
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        for i in range(len(perfect_models)):
            ssim_score = structural_similarity(
                perfect_models[i], segmented_3d_object, full=True)[0]
            
            if ssim_score > ssim_max_score:
                ssim_max_score = ssim_score
                ssim_max_score_index = i
        
        print("MAX SSIM SCORE:", ssim_max_score)
        print("PPM DEGREE OFFSET:", ppm_degree_offset[ssim_max_score_index])
        
        subtract = cv2.subtract(
            perfect_models[ssim_max_score_index], segmented_3d_object)
        
        CommonPrints.print_image("subtract", subtract, 600)
        
        cnts = CommonFunctionalities.find_and_grab_contours(subtract)
        
        filled_contours = np.zeros(masked_3d_object.shape, dtype=np.uint8)
        
        for c in cnts:
            if cv2.contourArea(c) > 200:
                cv2.fillPoly(filled_contours, [c], (251, 4, 131))
                
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    filled_contours, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        CommonPrints.print_image("Defect contours", filled_contours, 600)
        
        original_image_with_defects = cv2.add(
            masked_3d_object, filled_contours)
        
        CommonPrints.print_image(
            "Original image with defects", 
            original_image_with_defects, 
            600)
        
        return (
            original_image_with_defects, 
            ssim_max_score_index, 
            ssim_max_score)