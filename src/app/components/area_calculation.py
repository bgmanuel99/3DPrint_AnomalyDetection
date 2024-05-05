import os
import sys
import cv2
import imutils
import numpy as np

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.common.common import CommonPrints, CommonFunctionalities

class AreaCalculation(object):
    
    @classmethod
    def calculate_areas(cls, masked_3d_object: np.ndarray):
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        cnts, hierarchy = cv2.findContours(
            segmented_3d_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(len(cnts[0]))
        print(len(hierarchy[0][2]))
        
        infill_contours = np.zeros(
            masked_3d_object.shape, dtype=np.uint8)
        
        for c, level in zip(cnts, hierarchy):
            cv2.drawContours(infill_contours, [c], -1, (255, 255, 255), 1)
            break
            
        CommonPrints.print_image("infill contours", infill_contours, 600, True)