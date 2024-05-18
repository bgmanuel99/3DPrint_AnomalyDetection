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

FATHER_CLASSIFICATION_DIRECTORY_PATH: str = "/data/classification/"
TRAIN_IMAGES_DIRECTORY: str = "trainX/"
TRAIN_LABELS_DIRECTORY: str = "trainY/"
TRAIN_LABELS_FILE_NAME: str = "trainY.txt"
TEST_IMAGES_DIRECTORY: str = "testX/"
TEST_LABELS_DIRECTORY: str = "testY/"
TEST_LABELS_FILE_NAME: str = "testY.txt"

# ******************* Low contrast detection *******************
FRACTION_THRESHOLD: float = 0.35

# ******************* Gcode analizer *******************
GCODE_COMMENT_SYMBOL: str = ";"
GCODE_EXTRUSION_SYMBOL: str = "E"
# TODO: The feed rate could change, the only constant should be the F
GCODE_FEED_RATE_SYMBOL: str = "F9000" # Traveling extruder speed
GCODE_POSITION_SYMBOLS: tuple[str, str, str, str] = [
    "G1", "X", "Y", GCODE_EXTRUSION_SYMBOL]
GCODE_INITIAL_POSITION_SYMBOLS: tuple[str, str, str] = ["G1", "X", "Y"]
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

# ******************* Load data *******************
OUTPUT_DIRECTORY_PATH: str = "/data/output/"
OUTPUT_IMAGE_FILE_EXTENSION: str = "jpg"
OUTPUT_REPORT_EXTENSION: str = "pdf"

# ******************* Siamese neural network *******************
IMAGE_SHAPE: tuple[int] = (159, 120, 3)
BATCH_SIZE: int = 64
EPOCHS: int = 10
MODEL_PATH: str = "/data/classification/models/"
MODEL_NAME: str = "siamese_neural_network_model.keras"
PLOT_NAME: str = "siamese_model_plot.png"
RECOMENDATIONS_PATH = "/data/classification/recomendations/"
BAD_MATERIAL_ADHESION_FILE_NAME = "bad_material_adhesion.txt"
LOW_Z_OFFSET_FILE_NAME = "low_z_offset.txt"
STAINS_FILE_NAME = "stains.txt"
STRAND_WEAR_FILE_NAME = "strand_wear.txt"