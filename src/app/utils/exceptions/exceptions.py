# TODO: Separar clases de excepciones en diferentes ficheros por etapa del pipeline

class ImageDirectoryNotFoundException(Exception):
    """Raised when the image input directory is not found.

    Parameters:
        message --> Explanation message of the error
    """
    
    def __init__(self, message="Can't find input directory for images"):
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "ImageDirectoryNotFoundException: {}. Check data folder, there should be a ../input/image/ directory".format(self.message)
    
class GCodeDirectoryNotFoundException(Exception):
    """Raised when the gcode input directory is not found.

    Parameters:
        message --> Explanation message of the error
    """
    
    def __init__(self, message="Can't find input directory for gcode files"):
        self.message = message
        
        super().__init__(self.message)
    
    def __str__(self):
        return "GCodeDirectoryNotFoundException: {}. Check data folder, there should be a ../input/gcode/ directory".format(self.message)

class ExtractImageException(Exception):
    """Raised when the input image cannot be found.

    Parameters:
        image_name  --> The name of the image given by the user
        message     --> Explanation message of the error
    """
    
    def __init__(self, image_name, message="The image could not be found or is in an incorrect folder"):
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "ExtractImageException: image name -> {}. {}".format(self.image_name, self.message)
    
class ImageNotFileException(Exception):
    """Raised when the input image is not a file.

    Parameters:
        image_name  --> The name of the image given by the user
        message     --> Explanation message of the error
    """
    
    def __init__(self, image_name, message="The input image is not a valid file. Check the extension."):
        self.image_name = image_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "ImageNotFileException: image name -> {}. {}".format(self.image_name, self.message)
    
class ExtractGCodeFileException(Exception):
    """Raised when the input gcode file cannot be found.

    Parameters:
        file_name   --> The name of the gcode file given by the user
        message     --> Explanation message of the error
    """
    
    def __init__(self, file_name, message="The gcode file could not be found or is in an incorrect folder"):
        self.file_name = file_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "ExtractGCodeFileException: file name -> {}. {}".format(self.file_name, self.message)
    
class GCodeNotFileException(Exception):
    """Raised when the input gcode is not a file.

    Parameters:
        file_name  --> The name of the gcode file given by the user
        message    --> Explanation message of the error
    """
    
    def __init__(self, file_name, message="The input gcode is not a valid file. Check the extension."):
        self.file_name = file_name
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "GCodeNotFileException: file name -> {}. {}".format(self.file_name, self.message)

class LowContrastDetectionException(Exception):
    """Raised when an image is low contrast and cannot be used in the pipeline.
    
    Parameters:
        message --> Explanation message of the error
    """
    
    def __init__(self, message="The image has not enough contrast, please introduce a high contrast image to the pipeline"):
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self):
        return "LowContrastDetectionException: {}".format(self.message)