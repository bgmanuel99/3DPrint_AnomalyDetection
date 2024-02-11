import io

class GCodeAnalizer(object):
    
    """This class contains methods to analize a gcode file and extract the vertices and material extrusion data points
    
    Methods:
        extract_vertices (gcode_file: io.TextIOWrapper):
            Method to extract the information from the gcode file
    """
    
    @classmethod
    def extract_vertices(cls, gcode_file: io.TextIOWrapper):
        """Method to extract the information from the gcode file

        Parameters:
            gcode_file (io.TextIOWrapper): File with gcode information
        """
        
        for line in gcode_file.readlines():
            pass