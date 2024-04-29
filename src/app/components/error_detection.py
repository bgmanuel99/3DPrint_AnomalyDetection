import os
import sys
import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from app.common.common import print_image, print_images

class ErrorDetection(object):
    
    @classmethod
    def detect_errors(cls, segmented_image, perfect_model):
        print_image("segmented", segmented_image, 600)
        print_image("perfect model", perfect_model, 600)
        
        segmented_image_for_sub = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(perfect_model, segmented_image_for_sub)
        
        print_image("subtract", sub, 600)
        
        thresh = cv2.threshold(
            sub, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        print_image("subtract thresh", thresh, 600)
                
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        print_image("subtract thresh opening", opening, 600)
        
        #Detecting blobs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByCircularity = False

        im = cv2.bitwise_not(opening)
        
        print_image("im", im, 600)

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(im)

        #Drawing circle around blobs
        im_with_keypoints = cv2.drawKeypoints(segmented_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #Display image with circle around defect
        print_image("im_with_keypoints", im_with_keypoints, 600)
        
        overlay = perfect_model.copy()
        output = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
        
        print_image("Image Alignment Overlay", output)
        
        cls.detect_error(perfect_model, segmented_image)
        
    @classmethod
    def detect_error(cls, imageA, imageB):
        # convert the images to grayscale
        grayA = imageA
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        print(len(cnts))
        
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # show the output images
        print_images(
            ["Original", "Modified", "Diff", "Thresh"], 
            [imageA, imageB, diff, thresh], 
            600)