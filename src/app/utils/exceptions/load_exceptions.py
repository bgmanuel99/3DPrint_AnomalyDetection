class OutputImageDirectoryNotFound(Exception):
    """Raised when the image output directory is not found.

    Parameters:
        message (str): Explanation message of the error
    """
    
    def __init__(
            self, 
            message="Can't find output directory for images") -> None:
        self.message = message
        
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return (
            "OutputImageDirectoryNotFound: {}. "
            "Check data folder, there should be a ../output/ directory."
        ).format(self.message)