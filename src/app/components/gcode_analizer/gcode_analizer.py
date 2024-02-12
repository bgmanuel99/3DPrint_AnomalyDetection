import io
from typing import List

class GCodeAnalizer(object):
    
    """This class contains methods to analize a gcode file and extract the vertices and material extrusion data points
    
    Methods:
        extract_vertices (gcode_file: io.TextIOWrapper):
            Method to extract the information from the gcode file
    """
    
    @classmethod
    def extract_vertices(cls, gcode_file: io.TextIOWrapper) -> List[List]:
        """Method to extract the information from the gcode file

        Parameters:
            gcode_file (io.TextIOWrapper): File with gcode information
        """
        
        vertices = []
        
        for line in gcode_file.readlines():
            if (not line.startswith(";") 
                    and all(char in line for char in ["G1", "X", "Y", "E"])):
                gcode_position = line \
                    .strip() \
                    .replace(("G1", "X", "Y", "E"), ("", "", "", "")) \
                    .split()
                    
                vertices.append(list(map(int, gcode_position)))
        
        return vertices