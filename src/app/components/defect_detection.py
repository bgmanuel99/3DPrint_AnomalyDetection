import cv2
import numpy as np
from typing import List
from skimage.metrics import structural_similarity

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
            ppm_degree_offset: List[float]) -> tuple[
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
            ppm_degree_offset (List[float]):
                A list of the pixels per metric values containing the real 
                extracted value and others with a slight increase and decrease 
                value representing the degree error when the image of the real
                3d printed object is taken

        Returns:
            tuple[np.ndarray, int, float, float, float]:
                - A tuple with the image of the real 3d printed object with 
                the detected defectss
                - The max SSIM score index which will indicate the index in 
                the list of perfect models to know which of them was finally 
                used for the detection of the defects
                - The max SSIM score
                - The impresion defects error based on SSIM score
                - The segmentation defects error based on SSIM score
        """
        
        ssim_max_score = 0
        ssim_max_score_index = 0
        
        CommonPrints.print_image(
            "masked_3d_object", masked_3d_object, 600, True)
        
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        CommonPrints.print_image(
            "segmented_3d_object", segmented_3d_object, 600, True)
        
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
        
        CommonPrints.print_image("subtract", subtract, 600, True)
        
        cnts = CommonFunctionalities.find_and_grab_contours(subtract)
        
        impresion_defects_contours = np.zeros(
            masked_3d_object.shape, dtype=np.uint8)
        
        segmentation_defects_contours = np.zeros(
            masked_3d_object.shape, dtype=np.uint8)
        
        for c in cnts:
            if cv2.contourArea(c) > 200:
                cv2.fillPoly(impresion_defects_contours, [c], (255, 255, 255))
            else:
                cv2.fillPoly(
                    segmentation_defects_contours, [c], (255, 255, 255))
              
        # Calculate defects error based on ssim max score  
        impresion_defects_white_pixels = np.sum(
            impresion_defects_contours == 255)
        
        segmentation_defects_white_pixels = np.sum(
            segmentation_defects_contours == 255)
        
        total_ssim_diff = 1 - ssim_max_score
        
        total_white_pixels = (impresion_defects_white_pixels 
                              + segmentation_defects_white_pixels)
        
        impresion_defects_total_diff = (impresion_defects_white_pixels 
                                         * total_ssim_diff 
                                         / total_white_pixels)
        
        segmentation_defects_total_diff = (segmentation_defects_white_pixels 
                                            * total_ssim_diff 
                                            / total_white_pixels)
        
        # Change color of defects
        impresion_defects_contours[
            np.all(impresion_defects_contours == (255, 255, 255), axis=-1)
        ] = (251, 4, 131)
        
        segmentation_defects_contours[
            np.all(segmentation_defects_contours == (255, 255, 255), axis=-1)
        ] = (255, 0, 0)

        # Add defects bounding rectangles
        for c in cnts:
            if cv2.contourArea(c) > 200:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    impresion_defects_contours, 
                    (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 
                    1)
            else:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    segmentation_defects_contours, 
                    (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 
                    1)
        
        CommonPrints.print_image(
            "Impresion defects contours", 
            impresion_defects_contours, 
            600, 
            True)
        CommonPrints.print_image(
           "Segmentation defects contours", 
           segmentation_defects_contours, 
           600, 
           True)
        
        # Add original 3d object with the 3d impresion defects
        masked_3d_object_with_defects = cv2.add(
            masked_3d_object, impresion_defects_contours)
        
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