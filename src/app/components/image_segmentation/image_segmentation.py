import cv2 as cv
import matplotlib.pyplot as plt

class ImageSegmetation(object):
    
    @classmethod
    def segment_image(cls, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Convert the image to grayscale and blur it slightly
        # Blurring the image helps remove some of the high frequency edges in the image and allow a more clean segmentation
        
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
        
        ret, segmented = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        
        plt.imshow(segmented, cmap="gray")
        plt.axis('off')  # Turn off axis
        plt.show()
        
        masked = cv.bitwise_and(image, image, mask=segmented)
        
        plt.imshow(masked, cmap="gray")
        plt.axis('off')  # Turn off axis
        plt.show()