import os
import sys
import cv2
import copy
import imutils
import numpy as np
from imutils import contours
from skimage.metrics import structural_similarity

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.common.common import print_image

class ErrorDetection(object):
    
    @classmethod
    def detect_errors(
            cls, 
            segmented_image, 
            perfect_models, 
            ppm_degree_offset):
        
        ssim_max_score = 0
        ssim_max_score_index = 0
        
        gray_segmented_image = cv2.cvtColor(
            segmented_image, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(
            gray_segmented_image, 
            0, 
            255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        for i in range(len(perfect_models)):
            ssim_score = structural_similarity(
                perfect_models[i], thresh, full=True)[0]
            
            if ssim_score > ssim_max_score:
                ssim_max_score = ssim_score
                ssim_max_score_index = i
        
        print("MAX SSIM SCORE:", ssim_max_score)
        print("PPM DEGREE OFFSET:", ppm_degree_offset[ssim_max_score_index])
        
        sub = cv2.subtract(
            perfect_models[ssim_max_score_index], thresh)
        
        print_image("subtract", sub, 600, True)
        
        thresh = cv2.threshold(
            sub, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        print_image("subtract thresh", thresh, 600, True)
        
        #Detecting blobs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByCircularity = False

        im = cv2.bitwise_not(thresh)
        
        print_image("im", im, 600, True)

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(im)

        #Drawing circle around blobs
        im_with_keypoints = cv2.drawKeypoints(
            segmented_image, 
            keypoints,
            np.array([]), 
            (0, 0, 255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #Display image with circle around defect
        print_image("im_with_keypoints", im_with_keypoints, 600, True)
        
        cnts = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        filled_contours = np.zeros(segmented_image.shape, dtype=np.uint8)
        
        for c in cnts:
            if cv2.contourArea(c) > 200:
                cv2.fillPoly(filled_contours, [c], (251, 4, 131))
        
        print_image("Error contours", filled_contours, 600, True)
        
        original_image_with_errors = cv2.add(
            segmented_image, filled_contours)
        
        print_image(
            "Original image with errors", 
            original_image_with_errors, 
            600, 
            True)
        
    @classmethod
    def detect_error(cls, imageA, imageB):
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        return 