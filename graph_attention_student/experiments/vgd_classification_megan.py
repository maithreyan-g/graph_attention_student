import os
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.util import Skippable
from pycomex.experiment import SubExperiment
from graph_attention_student.models import Megan
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import mse
from graph_attention_student.training import NoLoss
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

PATH = pathlib.Path(__file__).parent.absolute()

TRAIN_RATIO = 0.8

# == MODEL PARAMETERS ==
UNITS: t.List[int] = [20, 20, 20]
DROPOUT_RATE: float = 0.0
IMPORTANCE_FACTOR: float = 1.0
IMPORTANCE_MULTIPLIER: float = 0.5
SPARSITY_FACTOR: float = 5.0
FINAL_UNITS: t.List[int] = [20, 10, 2]
FINAL_ACTIVATION: str = 'softmax'
FINAL_DROPOUT_RATE: float = 0.0
CONCAT_HEADS: bool = False

# == TRAINING PARAMETERS ==
EPOCHS: int = 20
BATCH_SIZE: int = 32
OPTIMIZER_CB: t.Callable = lambda: ks.optimizers.Adam(learning_rate=0.001)

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_classification.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('create_model')
    def create_model(e):
        e.info('MEGAN model')
        model = Megan(
            units=e.parameters['UNITS'],
            dropout_rate=e.parameters['DROPOUT_RATE'],
            importance_factor=e.parameters['IMPORTANCE_FACTOR'],
            importance_multiplier=e.parameters['IMPORTANCE_MULTIPLIER'],
            importance_channels=e.parameters['IMPORTANCE_CHANNELS'],
            sparsity_factor=e.parameters['SPARSITY_FACTOR'],
            final_units=e.parameters['FINAL_UNITS'],
            final_dropout_rate=e.parameters['FINAL_DROPOUT_RATE'],
            final_activation=e.parameters['FINAL_ACTIVATION'],
            use_graph_attributes=False,
            use_edge_features=True,
            concat_heads=e.parameters['CONCAT_HEADS']
        )
        model.compile(
            loss=[
                ks.losses.CategoricalCrossentropy(),
                NoLoss(),
                NoLoss(),
            ],
            loss_weights=[
                1, 0, 0
            ],
            metrics=[ks.metrics.CategoricalAccuracy()],
            optimizer=e.parameters['OPTIMIZER_CB'](),
            run_eagerly=False
        )
        return model


    @se.hook('fit_model')
    def fit_model(e, model, x_train, y_train, x_test, y_test):
        history = model.fit(
            x_train,
            y_train,
            batch_size=e.parameters['BATCH_SIZE'],
            epochs=e.parameters['EPOCHS'],
            validation_data=(x_test, y_test),
            validation_freq=1,
            callbacks=[
                LogProgressCallback(
                    logger=e.logger,
                    epoch_step=1,
                    identifier='val_output_1_categorical_accuracy'
                )
            ],
            verbose=0
        )
        return history.history


    @se.hook('query_model')
    def query_model(e, model, x, y, include_importances: bool = True):
        e.info('querying the model...')
        out_pred, ni_pred, ei_pred = model(x)

        if include_importances:
            return out_pred, ni_pred, ei_pred
        else:
            return out_pred


    @se.hook('calculate_fidelity')
    def calculate_fidelity(e, model, indices_true, x_true, y_true, out_pred, ni_pred, ei_pred):
        rep = e['rep']

        IMPORTANCE_CHANNELS = e.p['IMPORTANCE_CHANNELS']

        # ~ fidelity
        # For each importance channel we construct a mask which only masks out that very channel
        # and then we query the model using that mask, which effectively means that this channel
        # has absolutely no effect on the prediction. We record the outputs generated by these
        # masked predictions and then afterwards calculate the fidelity from that.
        for k in range(IMPORTANCE_CHANNELS):
            # First of all we need to construct the masks
            masks = []
            for ni in ni_pred:
                mask = np.ones_like(ni)
                mask[:, k] = 0
                masks.append(mask)

            masks_tensor = ragged_tensor_from_nested_numpy(masks)
            out_masked, _, _ = [v.numpy() for v in
                                model(x_true, node_importances_mask=masks_tensor)]

            for c, index in enumerate(indices_true):
                e[f'out/pred/{rep}/{index}'] = out_pred[c]
                e[f"out/masked/{rep}/{index}/{k}"] = out_masked[c]

        fidelities = []
        for index in indices_true:

            fidelity = 0
            for k in range(IMPORTANCE_CHANNELS):
                out = e[f"out/pred/{rep}/{index}"]
                out_masked = e[f"out/masked/{rep}/{index}/{k}"]
                fidelity += (out[k] - out_masked[k])

            fidelities.append(fidelity)

        return fidelities
