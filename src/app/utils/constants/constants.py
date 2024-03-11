# ******************* Extract data *******************
input_gcode_directory_path: str = "/data/input/gcode/"
gcode_file_extension: str = "gcode"
input_image_directory_path: str = "/data/input/image/"
image_file_extension: str = "png"

# ******************* Low contrast detection *******************
fraction_threshold: float = 0.35

# ******************* Gcode analizer *******************
gcode_comment_symbol: str = ";"
gcode_position_symbols: str = ["G1", "X", "Y", "E"]
gcode_initial_position_symbols: str = ["G1", "X", "Y", "F9000"]
wipe_start_comment: str = ";WIPE_START"
wipe_end_comment: str = ";WIPE_END"
retract_length_comment: str = "; retract_length ="
perimeter_types: str = [
    ";TYPE:External perimeter",
    ";TYPE:Internal infill"
]
layer_change_comment: str = ";LAYER_CHANGE"
z_value_comment: str = ";Z:"
height_value_comment: str = ";HEIGHT:"
input_strand: float = 2.404
layer_height: float = 0.26

# ******************* Load data *******************
output_image_directory_path: str = "/data/output/"
output_image_file_extension: str = "png"