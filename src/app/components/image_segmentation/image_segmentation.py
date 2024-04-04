import imutils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

class ImageSegmetation(object):
    
    @classmethod
    def segment_image(cls, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Convert the image to grayscale and blur it slightly
        # Blurring the image helps remove some of the high frequency edges in the image and allow a more clean segmentation
        
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        blurred_image = cv.GaussianBlur(gray_image, (7, 7), 0)
        
        _, segmented = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        
        plt.imshow(segmented, cmap="gray")
        plt.axis('off')  # Turn off axis
        plt.show()
        
        # Find the contour for the reference object in the threshold image
        cnts = cv.findContours(segmented, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts, method="bottom-to-top")
        
        cnts = [c for c in cnts if cv.contourArea(c) > 1000]
        
        pixelsPerMetric = cls.get_pixels_per_metric(cnts)
        
        external_contour = np.zeros(image.shape[0:2], dtype=np.uint8)
        filled_contour = external_contour.copy()
        
        cv.drawContours(external_contour, [cnts[1]], -1, (255, 255, 255), 2)
        
        cv.fillPoly(filled_contour, [cnts[1]], (255, 255, 255))
        
        full_contour = external_contour + filled_contour
        
        plt.imshow(full_contour, cmap="gray")
        plt.axis('off')  # Turn off axis
        plt.show()
        
        masked = cv.bitwise_and(image, image, mask=full_contour)
        masked = cv.bitwise_and(masked, masked, mask=segmented)
        
        plt.imshow(masked, cmap="gray")
        plt.axis('off')  # Turn off axis
        plt.show()
        
        return masked, pixelsPerMetric
    
    @classmethod
    def mid_point(cls, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
    @classmethod
    def get_pixels_per_metric(cls, cnts):
        box = cv.minAreaRect(cnts[0])
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
    
        (tl, tr, br, bl) = box
        
        (tlblX, tlblY) = cls.mid_point(tl, bl)
        (trbrX, trbrY) = cls.mid_point(tr, br)
        
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        return dB / 1.874