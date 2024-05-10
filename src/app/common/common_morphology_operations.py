import cv2
import numpy as np

class CommonMorphologyOperations(object):
    """Class contining common functions for morphology operations over images
    
    Methods:
        morphologyEx_opening (
                segmented_image: np.ndarray, 
                kernel_size: tuple[float, float]):
            Method to make an opening morphology operation over an image
    """
    
    @staticmethod
    def morphologyEx_opening(
            segmented_image: np.ndarray, 
            kernel_size: tuple[float, float]) -> np.ndarray:
        """Method to make an opening morphology operation over an image

        Parameters:
            segmented_image (np.ndarray): 
                Image to which the operation is performed
            kernel_size (tuple[float, float]): 
                Kernel size of the opening operation

        Returns:
            np.ndarray: Image with the opening operation
        """
        
        kernel = np.ones(kernel_size, np.uint8)
        
        opening = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
        
        return opening