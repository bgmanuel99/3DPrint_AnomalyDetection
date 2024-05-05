import cv2
import math
import numpy as np
from imutils import contours

from app.common.common import CommonPrints, CommonFunctionalities, CommonMorphologyOperations

class AreaCalculation(object):

    @classmethod
    def calculate_areas(
            cls, 
            masked_3d_object: np.ndarray, 
            reference_object_width: float, 
            reference_object_pixels_area: float):
        
        segmented_3d_object = CommonFunctionalities.get_segmented_image(
            masked_3d_object)
        
        opening = CommonMorphologyOperations.morphologyEx_opening(
            segmented_3d_object, (5, 5))
        
        CommonPrints.print_image("opening", opening, 600, True)

        cnts, _ = cv2.findContours(
            opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from max to min by area
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]

        # Sort contours from left to right and top to bottom
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")
        cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")

        infill_contours = np.zeros(
            masked_3d_object.shape, dtype=np.uint8)
        
        infill_pixels_areas = []

        for (i, c) in enumerate(cnts):
            M = cv2.moments(c)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            cv2.drawContours(infill_contours, [c], -1, (255, 255, 255), 1)
            cv2.putText(
                infill_contours, 
                "{}".format(i+1), 
                (center_x-10, center_y+10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                1)
            
            infill_pixels_areas.append([i+1, cv2.contourArea(c)])

        CommonPrints.print_image("infill contours", infill_contours, 600, True)
        
        # Calculate area in mm2 of the reference object
        reference_object_area = math.pow(reference_object_width/2, 2) * math.pi
        
        infill_areas = []
        
        for i, infill_pixels_area in infill_pixels_areas:
            infill_areas.append([
                i, 
                infill_pixels_area 
                * reference_object_area 
                / reference_object_pixels_area])
            
        return infill_contours, infill_areas