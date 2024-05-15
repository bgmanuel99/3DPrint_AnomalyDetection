import io
import math
from typing import List
from scipy.spatial import distance as dist

from app.utils.constants.constants import *

class GCodeAnalizer(object):
    """This class contains methods to analize a gcode file and extract all the 
    needed information to create the perfect model of the 3d printed object
    
    Methods:
        extract_data (gcode_file: io.TextIOWrapper):
            Method to extract the information from the gcode file
        _extract_structured_data (gcode_file: io.TextIOWrapper):
            Private method to analize and extract the needed data from the 
            gcode file
        _calculate_extrusion_relative_distances:
            Private method to calculate the relative distances for the 
            extrusion data
        _calculate_width_from_relative_extrusion:
            Private method to calculate the filament widths based on the 
            relative extrusion and the coord distances
        _calculate_output_strand (
                cls,
                extrusion_data: float,
                coords_distance: float):
            Private method to calculate the output strand of the filament
        _calculate_width(
                cls,
                output_strand: float):
            Private method to calculate the width of the filament
        _is_new_layer (
                line: str, 
                first_layer: bool, 
                layer: int,
                first_layer_type: bool):
            Private method to check if the line describes the initiation of a 
            new layer in the gcode file
        _is_z_value (line: str, layer: int):
            Private method to check if the line contains the z value of the 
            actual layer
        _is_height_value (line: str, layer: int):
            Private method to check if the line contains the height value of 
            the actual layer
        _is_retract_length (line: str):
            Private method to check if the line contains the retract length 
            value for the gcode file
        _is_perimeter_type (
                line: str, 
                layer: int, 
                first_layer_type: bool, 
                layer_type: int, 
                initial_coords: List[float]):
            Private method to check if it's a new perimeter type in the gcode 
            file for the actual layer
        _is_initial_coord (
                line: str, 
                wiping: bool,
                initial_coords: List[float]):
            Private method to check if it's an initial coord in the gcode file
        _is_new_coord (
                line: str, 
                layer: int, 
                layer_type: int, 
                wiping: bool):
            Private method to check if it's a new coord in the gcode file
    """
    
    _coords: List[List[object]] = []
    _retract_length: float = 0.0
    
    @classmethod
    def extract_data(
            cls, 
            gcode_file: io.TextIOWrapper) -> List[List[object]]:
        """Method to extract the information from the gcode file and transform 
        it to create the perfect model

        Parameters:
            gcode_file (io.TextIOWrapper): File with gcode information
            
        Returns:
            List[List[object]]: Real coordinates of the 3d printed object
        """
        
        cls._extract_structured_data(gcode_file)
        
        cls._calculate_extrusion_relative_distances()
        
        cls._calculate_width_from_relative_extrusion()
        
        return cls._coords
    
    @classmethod
    def _extract_structured_data(cls, gcode_file: io.TextIOWrapper) -> None:
        """Method to analize and extract the needed data from the gcode file

        Parameters:
            gcode_file (io.TextIOWrapper): File with gcode information
        """
        
        wiping: bool = False
        first_layer: bool = True
        layer: int = 0
        first_layer_type: bool = True
        layer_type: int = 0
        initial_coords: List[float] = []
        
        for line in gcode_file.readlines():
            # Check new layer
            (first_layer, 
             layer, 
             first_layer_type, 
             is_new_layer) = cls._is_new_layer(
                line, first_layer, layer, first_layer_type)
            
            if is_new_layer: 
                continue
            
            # Check for initial coords
            initial_coords, is_initial_coords = cls._is_initial_coord(
                line, wiping, initial_coords)
            
            # Check new perimeter
            (first_layer_type, 
             layer_type, 
             is_new_perimeter) = cls._is_perimeter_type(
                line, layer, first_layer_type, layer_type, initial_coords)
            
            if is_initial_coords:
                continue
            elif is_new_perimeter:
                continue
            # Check Z value
            elif cls._is_z_value(line, layer): 
                continue
            # Check height value
            elif cls._is_height_value(line, layer):
                continue
            # Get retract length value
            elif cls._is_retract_length(line):
                continue
            # Pass by any coord position between wipe comments
            elif line.strip() == WIPE_START_COMMENT: 
                wiping = True
                continue
            elif line.strip() == WIPE_END_COMMENT: 
                wiping = False
                continue
            # Check new coords
            elif cls._is_new_coord(line, layer, layer_type, wiping):
                continue
        
        gcode_file.close()
        
    @classmethod
    def _calculate_extrusion_relative_distances(cls) -> None:
        """Method to calculate the relative distances for the extrusion data
        """
        
        # For each layer of the gcode file
        for layer in cls._coords:
            # For each perimeter that comprehends a layer of the gcode file
            for perimeter in layer[2]:
                for i in range(len(perimeter[1])-1, -1, -1):
                    # If there is no extrusion data for the coord of the 
                    # perimeter, it means it's an initial coord only for 
                    # positioning
                    if perimeter[1][i][2] == 0.0:
                        continue
                    else:
                        # If the next coord for the actual one is the 
                        # initial coord then use the retract length to 
                        # calculate relative distance
                        if perimeter[1][i-1][2] == 0.0:
                            if perimeter[1][i][2] > cls._retract_length: 
                                perimeter[1][i][2] = round(perimeter[1][i][2] 
                                                        - cls._retract_length,
                                                        2)
                        else:
                            # If there are two coords with extrusion data use
                            # both to calculate the relative distance
                            perimeter[1][i][2] = round(perimeter[1][i][2] 
                                                       - perimeter[1][i-1][2], 
                                                       2)
                            
    @classmethod
    def _calculate_width_from_relative_extrusion(cls) -> None:
        """Method to calculate the filament widths based on the relative 
        extrusion and the coord distances
        """
        
        # For each layer of the gcode file
        for layer in cls._coords:
            # For each perimeter that comprehends a layer of the gcode file
            for perimeter in layer[2]:
                for i in range(len(perimeter[1])):
                    if perimeter[1][i][2] == 0.0:
                        continue
                    else:
                        distance = dist.euclidean(
                            (perimeter[1][i-1][0], perimeter[1][i-1][1]),
                            (perimeter[1][i][0], perimeter[1][i][1]))
                        
                        output_strand = cls._calculate_output_strand(
                            perimeter[1][i][2],
                            distance)
                        
                        perimeter[1][i][2] = cls._calculate_width(
                            output_strand)
    
    @classmethod
    def _calculate_output_strand(
            cls,
            extrusion_data: float,
            coords_distance: float) -> float:
        """Method to calculate the output strand of the filament. The formula 
        to calculate it is:
            ouput_strand = input_strand * extrusion_data / coords_distance

        Parameters:
            extrusion_data (float): Relative extrusion data
            coords_distance (float): Distance between two coords

        Returns:
            float: Output strand
        """
        
        return INPUT_STRAND * extrusion_data / coords_distance
    
    @classmethod
    def _calculate_width(
            cls,
            output_strand: float) -> float:
        """Method to calculate the width of the filament. The formula to 
        calculate it is:
            output_strand = layer_height * (layer_width - layer_height) + 
                Pi * (layer_height / 2) ^ 2
        Solving the width of the equation:
            width = ((output_strand - Pi * (layer_height / 2) ^ 2) / 
                layer_height) + layer_height

        Parameters:
            output_strand (float): Output strand size

        Returns:
            float: Filament width
        """
        
        return round((output_strand - math.pi * math.pow(LAYER_HEIGHT / 2, 2)) 
                / LAYER_HEIGHT 
                + LAYER_HEIGHT, 
                2)
        
    @classmethod
    def _is_new_layer(
            cls, 
            line: str, 
            first_layer: bool, 
            layer: int,
            first_layer_type: bool) -> tuple[bool, int, bool, bool]:
        """Method to check if the line describes the initiation of a new layer 
        in the gcode file

        Parameters:
            line (str): Gcode file line
            first_layer (bool): Boolean to know if it's first layer
            layer (int): Number of the layer
            first_layer_type (bool): 
                Boolean to know if it's the first type of perimeter in the 
                actual layer

        Returns:
            tuple[bool, int, bool, bool]: 
                First layer, layer, first layer type and boolean to 
                acknowledge if it is or not a new layer
        """
        
        if line.strip() == LAYER_CHANGE_COMMENT:
            if not first_layer: 
                layer += 1
            
            first_layer = False
            cls._coords.append([])
            
            return first_layer, layer, True, True
            
        return first_layer, layer, first_layer_type, False
    
    @classmethod
    def _is_z_value(cls, line: str, layer: int) -> bool:
        """Method to check if the line contains the z value of the actual layer

        Parameters:
            line (str): Gcode file line
            layer (int): Number of the layer

        Returns:
            bool: Boolean to acknowledge if it is or not the z value
        """
        
        if Z_VALUE_COMMENT in line:
            z_value = float(line
                .strip()
                .split(":")[1])
            
            cls._coords[layer].append(z_value)
            
            return True
        
        return False
                
    @classmethod
    def _is_height_value(cls, line: str, layer: int) -> bool:
        """Method to check if the line contains the height value of the actual 
        layer

        Parameters:
            line (str): Gcode file line
            layer (int): Number of the layer

        Returns:
            bool: Boolean to acknowledge if it is or not the height value
        """
        
        if HEIGHT_VALUE_COMMENT in line:
            height_value = float(line
                .strip()
                .split(":")[1])
            
            cls._coords[layer].append(height_value)
            cls._coords[layer].append([])
            
            return True
        
        return False
    
    @classmethod
    def _is_retract_length(cls, line: str) -> bool:
        """Method to check if the line contains the retract length value for 
        the gcode file

        Parameters:
            line (str): Gcode file line

        Returns:
            bool: 
                Boolean to acknowledge if it is or not the retract length value
        """
        
        if RETRACT_LENGTH_COMMENT in line:
            cls._retract_length = float(line.split("=")[1].strip())
            
            return True
        
        return False
    
    @classmethod
    def _is_perimeter_type(
            cls, 
            line: str, 
            layer: int, 
            first_layer_type: bool, 
            layer_type: int, 
            initial_coords: List[float]) -> tuple[bool, int, bool]:
        """Method to check if it's a new perimeter type in the gcode file for 
        the actual layer

        Parameters:
            line (str): Gcode file line
            layer (str): Number of the layer
            first_layer_type (bool): 
                Boolean to know if it's the first type of perimeter in the 
                actual layer
            layer_type (int): Number of the perimeter
            initial_coords (List[float]): Initial coords for a perimeter
            
        Returns:
            tuple[bool, int, bool]: 
                First layer type, layer type and  boolean to acknowledge if it 
                is a new perimeter type
        """
        
        for type in PERIMETER_TYPES:
            if line.strip() == type:
                if not first_layer_type: 
                    layer_type += 1
                    
                first_layer_type = False
                type_name = type.split(":")[1]
                cls._coords[layer][2].append([type_name, [initial_coords]])
                
                return first_layer_type, layer_type, True
        
        return first_layer_type, layer_type, False
    
    @classmethod
    def _is_initial_coord(
            cls, 
            line: str, 
            wiping: bool,
            initial_coords: List[float]) -> tuple[List[float], bool]:
        """Method to check if it's an initial coord in the gcode file

        Parameters:
            line (str): Gcode file line
            wiping (bool): Boolean to know if the coord should be dismissed
            initial_coords (List[float]): Initial coords for a perimeter

        Returns:
            tuple[List[float], bool]: 
                List with the inital coords and boolean to acknowledge it is a 
                initial coord
        """
        
        if (all([char in line for char in GCODE_INITIAL_POSITION_SYMBOLS])
                and GCODE_EXTRUSION_SYMBOL not in line
                and not wiping
                and GCODE_COMMENT_SYMBOL not in line
                and not line.startswith(GCODE_COMMENT_SYMBOL)):
            if GCODE_FEED_RATE_SYMBOL in line:
                gcode_position = (line
                    .strip()
                    .replace(GCODE_INITIAL_POSITION_SYMBOLS[0], "")
                    .replace(GCODE_INITIAL_POSITION_SYMBOLS[1], "")
                    .replace(GCODE_INITIAL_POSITION_SYMBOLS[2], "")
                    .replace(GCODE_FEED_RATE_SYMBOL, "0")
                    .split())
            else:
                gcode_position = (line
                    .strip()
                    .replace(GCODE_INITIAL_POSITION_SYMBOLS[0], "")
                    .replace(GCODE_INITIAL_POSITION_SYMBOLS[1], "")
                    .replace(GCODE_INITIAL_POSITION_SYMBOLS[2], "")
                    .split())
                gcode_position.append(0.0)
            
            initial_coords = list(map(float, gcode_position))
            
            return initial_coords, True
    
        return initial_coords, False
    
    @classmethod
    def _is_new_coord(
            cls, 
            line: str, 
            layer: int, 
            layer_type: int, 
            wiping: bool) -> bool:
        """Method to check if it's a new coord in the gcode file

        Parameters:
            line (str): Gcode file line
            layer (int): Number of the layer
            layer_type (int): Number of the perimeter
            wiping (bool): Boolean to know if the coord should be dismissed

        Returns:
            bool: Boolean to acknowledge it is a new coord
        """
        
        if (all([char in line for char in GCODE_POSITION_SYMBOLS])
                and not wiping
                and GCODE_COMMENT_SYMBOL not in line
                and not line.startswith(GCODE_COMMENT_SYMBOL)):
            gcode_position = (line
                .strip()
                .replace(GCODE_POSITION_SYMBOLS[0], "")
                .replace(GCODE_POSITION_SYMBOLS[1], "")
                .replace(GCODE_POSITION_SYMBOLS[2], "")
                .replace(GCODE_POSITION_SYMBOLS[3], "")
                .split())
            
            cls._coords[layer][2][layer_type][1].append(
                list(map(float, gcode_position)))
            
            return True
        
        return False