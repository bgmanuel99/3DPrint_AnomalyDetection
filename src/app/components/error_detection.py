import cv2 as cv
import matplotlib.pyplot as plt

class ErrorDetection(object):
    
    @classmethod
    def detect_errors(cls, segmented_image, perfect_model):
        masked = cv.bitwise_and(segmented_image, segmented_image, mask=perfect_model)
        
        plt.imshow(masked, cmap="gray")
        plt.show()