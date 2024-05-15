import os
from typing import List

# ******************* Extract data *******************
INPUTS_FATHER_DIRECTORY_PATH: str = "/data/input/"
INPUT_IMAGE_DIRECTORY: str = "image/"
INPUT_GCODE_DIRECTORY: str = "gcode/"
INPUT_METADATA_DIRECTORY: str = "metadata/"
GCODE_FILE_EXTENSION: str = "gcode"
IMAGE_FILE_EXTENSIONS: List[str] = [
    "png", "jpg", "jpeg", "jpe", "bmp", "dib", "jp2", "pbm", "pgm", "ppm", 
    "sr", "ras", "tiff", "tif"]
METADATA_FILE_EXTENSION: str = "txt"

# ******************* Low contrast detection *******************
FRACTION_THRESHOLD: float = 0.35

# ******************* Gcode analizer *******************
GCODE_COMMENT_SYMBOL: str = ";"
GCODE_EXTRUSION_SYMBOL: str = "E"
# TODO: The feed rate could change, the only constant should be the F
GCODE_FEED_RATE_SYMBOL: str = "F9000" # Traveling extruder speed
GCODE_POSITION_SYMBOLS: tuple[str, ...] = [
    "G1", "X", "Y", GCODE_EXTRUSION_SYMBOL]
GCODE_INITIAL_POSITION_SYMBOLS: tuple[str, ...] = ["G1", "X", "Y"]
WIPE_START_COMMENT: str = ";WIPE_START"
WIPE_END_COMMENT: str = ";WIPE_END"
RETRACT_LENGTH_COMMENT: str = "; retract_length ="
PERIMETER_TYPES: str = [
    ";TYPE:External perimeter",
    ";TYPE:Internal infill"
]
LAYER_CHANGE_COMMENT: str = ";LAYER_CHANGE"
Z_VALUE_COMMENT: str = ";Z:"
HEIGHT_VALUE_COMMENT: str = ";HEIGHT:"
INPUT_STRAND: float = 2.404
LAYER_HEIGHT: float = 0.25

# ******************* Siamese neural network *******************
IMAGE_SHAPE: tuple[int] = (159, 120, 3)
BATCH_SIZE: float = 64
EPOCHS: int = 5
MODEL_PATH: str = (os.path.dirname(os.getcwd()) 
                   + "/data/classification/output/siamese_model.h5")
PLOT_PATH: str = (os.path.dirname(os.getcwd()) 
                  + "/data/classification/output/siamese_model_plot.png")

# ******************* Load data *******************
OUTPUT_DIRECTORY_PATH: str = "/data/output/"
OUTPUT_IMAGE_FILE_EXTENSION: str = "jpg"
OUTPUT_REPORT_EXTENSION: str = "pdf"