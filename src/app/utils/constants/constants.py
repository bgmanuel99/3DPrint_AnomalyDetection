from typing import List

# ******************* Extract data *******************
inputs_father_directory_path: str = "/data/input/"
input_image_directory: str = "image/"
input_gcode_directory: str = "gcode/"
input_metadata_directory: str = "metadata/"
gcode_file_extension: str = "gcode"
image_file_extensions: List[str] = [
    "png", "jpg", "jpeg", "jpe", "bmp", "dib", "jp2", "pbm", "pgm", "ppm", 
    "sr", "ras", "tiff", "tif"]
metadata_file_extension: str = "txt"

# ******************* Low contrast detection *******************
fraction_threshold: float = 0.35

# ******************* Gcode analizer *******************
gcode_comment_symbol: str = ";"
gcode_extrusion_symbol: str = "E"
# TODO: The feed rate could change, the only constant should be the F
gcode_feed_rate_symbol: str = "F9000" # Traveling extruder speed
gcode_position_symbols: str = ["G1", "X", "Y", gcode_extrusion_symbol]
gcode_initial_position_symbols: str = ["G1", "X", "Y"]
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
layer_height: float = 0.25

# ******************* Image generator *******************

# ******************* Load data *******************
output_directory_path: str = "/data/output/"
output_image_file_extension: str = "jpg"
output_report_extension: str = "pdf"