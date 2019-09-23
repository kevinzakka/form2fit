"""Config file.
"""

import os
import numpy as np


# directories
root_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(root_dir, "code")
ml_data_dir = os.path.join(root_dir, "ml/dataset/")
weights_dir = os.path.join(root_dir, "ml/models/weights")
results_dir = os.path.join(root_dir, "results")
train_dir = os.path.join(root_dir, "train")
logs_dir = os.path.join(root_dir, "logs")

# heightmap params
VIEW_BOUNDS = np.asarray([[-0.2, .728], [0.38, 1.1], [-1, 1]])
HEIGHTMAP_RES = 0.002

# baseline parameters
MIN_NUM_MATCH = 4

# background subtraction
BACKGROUND_SUBTRACT = {
    "black-floss": (0.04, 0.047),
    "tape-runner": (0.04, 0.047),
    "zoo-animals": None,
    "fruits": (0.04, 0.047),
    "deodorant": (0.04, 0.047),
}

# the watermelon fruit is 360-degree symmetrical
# thus we don't want to penalize rotational errors
FRUIT_IDXS = [0, 4, 8, 12, 16, 20]

# plotting params
MAX_LENGTH_THRESH = 0.1
MAX_DEG_THRESH = 200
MAX_PIX_THRESH = 30
MAX_TR_THRESH = 0.15