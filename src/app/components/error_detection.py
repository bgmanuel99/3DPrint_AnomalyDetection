import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.common.common import CommonPrints, CommonFunctionalities

class ErrorDetection(object):
    
    @classmethod
    def detect_errors(
            cls, 
            masked_3d_object, 
            perfect_models, 
            ppm_degree_offset):
        
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
        
        CommonPrints.print_image("subtract", subtract, 600, True)
        
        cnts = CommonFunctionalities.find_and_grab_contours(subtract)
        
        filled_contours = np.zeros(masked_3d_object.shape, dtype=np.uint8)
        
        for c in cnts:
            if cv2.contourArea(c) > 200:
                cv2.fillPoly(filled_contours, [c], (251, 4, 131))
                
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    filled_contours, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        CommonPrints.print_image("Error contours", filled_contours, 600, True)
        
        original_image_with_errors = cv2.add(
            masked_3d_object, filled_contours)
        
        CommonPrints.print_image(
            "Original image with errors", 
            original_image_with_errors, 
            600, 
            True)
        
    @classmethod
    def detect_error(cls, imageA, imageB):
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        return 