import typing as t

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.conv.schnet_conv import SchNetInteraction
from kgcnn.layers.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition, ShiftPeriodicLattice


class SchNet(ks.models.Model):

    def __init__(self,
                 interaction_units: int = 128,
                 interaction_activation: str = 'kgcnn>shifted_softplus',
                 dense_units: t.List[int] = [64],
                 output_units: t.List[int] = [64, 1],
                 output_activation: str = 'linear',
                 use_bias: bool = True,
                 gauss_bins: int = 20,
                 gauss_distance: int = 4,
                 gauss_offset: float = 0.0,
                 gauss_sigma: float = 0.4,
                 ):
        super(SchNet, self).__init__()
        self.interaction_units = interaction_units
        self.dense_units = dense_units
        self.output_units = output_units
        self.interaction_activation = interaction_activation
        self.use_bias = use_bias
        self.gauss_bins = gauss_bins
        self.gauss_distance = gauss_distance
        self.gauss_offset = gauss_offset
        self.gauss_sigma = gauss_sigma

        self.lay_node_pos = NodePosition()
        self.lay_distance = NodeDistanceEuclidean()
        self.lay_gauss_basis = GaussBasisLayer(
            bins=gauss_bins,
            distance=gauss_distance,
            offset=gauss_offset,
            sigma=gauss_sigma,
        )
        self.lay_dense = DenseEmbedding(
            units=interaction_units,
            activation='linear',
        )
        self.lay_interaction = SchNetInteraction(
            units=interaction_units,
            activation=interaction_activation,
            use_bias=use_bias,
        )

        # ~ prediction backend
        self.lay_pool = PoolingNodes(pooling_method='sum')
        self.output_layers = []
        self.output_activations = ['kgcnn>shifted_softplus' for _ in output_units]
        self.output_activations[-1] = output_activation
        for k, act in zip(self.output_units, self.output_activations):
            lay = DenseEmbedding(
                units=k,
                activation=act,
                use_bias=True,
            )
            self.output_layers.append(lay)

    def call(self,
             inputs,
             return_edge_embeddings: bool = False):
        node_input, xyz_input, edge_indices = inputs

        pos1, pos2 = self.lay_node_pos([xyz_input, edge_indices])
        edge_distance = self.lay_distance([pos1, pos2])
        edge_gauss = self.lay_gauss_basis(edge_distance)

        edge_embedding = self.lay_dense(edge_gauss)
        edge_embedding = self.lay_interaction([edge_embedding, edge_gauss, edge_indices])

        graph_embedding = self.lay_pool(edge_embedding)
        output = graph_embedding
        for lay in self.output_layers:
            output = lay(output)

        if return_edge_embeddings:
            return edge_embedding

        return output
