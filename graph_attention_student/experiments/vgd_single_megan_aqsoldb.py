import os
import pathlib
import typing as t

import tensorflow as tf
from tensorflow import keras as ks
# from pycomex.util import Skippable
# from pycomex.experiment import SubExperiment
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('C:\\Users\\maith\\Downloads\\aqsoldb\\aqsoldb')
USE_DATASET_SPLIT: t.Optional[int] = 0
TRAIN_RATIO: float = 0.8
NUM_EXAMPLES: int = 100
NUM_TARGETS: int = 1

NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False
USE_EDGE_ATTRIBUTES: bool = True

# == MODEL PARAMETERS ==
UNITS = [48, 48, 48]
DROPOUT_RATE = 0.1
FINAL_UNITS = [48, 32, 16, 1]
REGRESSION_REFERENCE = [[-2.0]]
REGRESSION_WEIGHTS = [[1.0, 1.0]]
IMPORTANCE_CHANNELS: int = 2
IMPORTANCE_FACTOR = 2.0
IMPORTANCE_MULTIPLIER = 0.6
FIDELITY_FACTOR = 0.2
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
SPARSITY_FACTOR = 0.1
CONCAT_HEADS = False

# == TRAINING PARAMETERS ==
BATCH_SIZE = 32
EPOCHS = 50
REPETITIONS = 1
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(learning_rate=0.001)

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_single_megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()
