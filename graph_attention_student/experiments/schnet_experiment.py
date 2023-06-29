import random
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.models.schnet import SchNet
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.models.megan import Megan
from graph_attention_student.training import NoLoss

__DEBUG__ = True

VISUAL_GRAPH_DATASET_PATH = '/home/jonas/.visual_graph_datasets/datasets/dipole_moment'
NUM_TEST = 100
NUM_CHANNELS = 2


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log("loading dataset...")
    reader = VisualGraphDatasetReader(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()  # example - 12: {'metadata': {'graph': ...}}}
    indices = list(index_data_map.keys())

    test_indices = random.sample(indices, k=NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))

    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
    graphs_train = [index_data_map[i]['metadata']['graph'] for i in train_indices]

    y_test = np.array([index_data_map[i]['metadata']['target'] for i in test_indices])
    y_train = np.array([index_data_map[i]['metadata']['target'] for i in train_indices])

    x_schnet_train = (
        ragged_tensor_from_nested_numpy([graph['node_attributes'] for graph in graphs_train]),
        ragged_tensor_from_nested_numpy([graph['node_coordinates'] for graph in graphs_train]),
        ragged_tensor_from_nested_numpy([graph['edge_indices'] for graph in graphs_train]),
    )

    schnet = SchNet()

    schnet.compile(
        optimizer=ks.optimizers.Adam(learning_rate=1e-3),
        loss=ks.losses.MeanSquaredError(),
    )

    schnet.fit(
        x_schnet_train, y_train,
        epochs=1,
        batch_size=16
    )

    schnet_embeddings = schnet(x_schnet_train, return_edge_embeddings=True)
    schnet_embeddings = schnet_embeddings.numpy()

    # Dataset Augmentation
    for graph, emb in zip(graphs_train, schnet_embeddings):

        graph['edge_attributes'] = np.concatenate([
            graph['edge_attributes'], emb
        ], axis=-1)

    x_megan_train = tensors_from_graphs(graphs_train)
    y_megan_train = (
        y_train,
        ragged_tensor_from_nested_numpy([np.zeros(shape=(len(graph['node_indices']), NUM_CHANNELS)) for graph in graphs_train]),
        ragged_tensor_from_nested_numpy([np.zeros(shape=(len(graph['edge_indices']), NUM_CHANNELS)) for graph in graphs_train]),
    )

    megan = Megan(
        units=[32, 32, 32],
        final_units=[32, 16, 1],
        importance_channels=NUM_CHANNELS,
        importance_factor=0.0,
    )
    megan.compile(
        optimizer=ks.optimizers.Adam(learning_rate=1e-3),
        loss=[
            ks.losses.MeanSquaredError(),
            NoLoss(),
            NoLoss(),
        ],
    )
    megan.fit(
        x_megan_train, y_megan_train,
        epochs=25,
        batch_size=16
    )


experiment.run_if_main()
