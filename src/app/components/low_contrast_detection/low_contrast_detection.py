import cv2 as cv

class LowContrastDetection:
    
    @staticmethod
    def low_contrast_dectection(image):
        # Change color channels from BGR -> RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Convert the image to grayscale
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        
        # Slightly blur the image and perform edge detection
        blur_image = cv.GaussianBlur(gray_image, (5, 5), 0)
        edged = cv.Canny(blur_image, 30, 150)