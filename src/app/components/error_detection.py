import cv2 as cv
import matplotlib.pyplot as plt

class ErrorDetection(object):
    
    @classmethod
    def detect_errors(cls, segmented_image, perfect_model):
        plt.imshow(segmented_image, cmap="gray")
        plt.show()
        
        plt.imshow(perfect_model, cmap="gray")
        plt.show()
        
        print(segmented_image.shape)
        print(perfect_model.shape)
        
        masked = cv.bitwise_and(segmented_image, segmented_image, mask=perfect_model)
        
        plt.imshow(masked, cmap="gray")
        plt.show()