"""
Base experiment for the training of classification models based on visual graph datasets.

CHANGELOG

0.1.0 - 29.03.2023 - Initial version
"""
import os
import sys
import pathlib
import random
import typing as t
from collections import Counter

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from pycomex.util import Skippable
from pycomex.experiment import Experiment
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

import graph_attention_student.typing as tc
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.util import array_normalize, binary_threshold
from graph_attention_student.util import latex_table, latex_table_element_mean
from graph_attention_student.util import render_latex

# == DATASET PARAMETERS ==
# These parameters define the dataset that is to be used for the experiment as well as properties of
# that dataset such as the train test split for example.

# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/rb_adv_motifs')
# If this is None, then a new random split will be created for every repetition of the experiment.
# Otherwise this may be the int index of an existing split that is associated with the dataset and then
# that split will be used instead all the time.
USE_DATASET_SPLIT: t.Optional[int] = None
# :param TRAIN_RATIO:
#       The percentage of the dataset to use for random train set
TRAIN_RATIO: float = 0.8
# :param NUM_CLASSES:
#       The number of classes represented in the dataset
NUM_CLASSES: int = 2
# :param CLASS_LABELS:
#       A list with the same number of elements as there are classes in the dataset, where each string
#       element is a short description/name for the class at the corresponding index.
CLASS_LABELS: t.List[str] = ['false motif', 'true motif']

# :param NUM_EXAMPLES:
#       THis is the number of elements from the test set which is going to be used as examples for the
#       visualization of the explanations generated by the model
NUM_EXAMPLES: int = 100
# :param EXAMPLE_INDICES:
#       This may be a list of dataset indices. The corresponding elements will always be part of the example
#       set and be visualized. This feature would be important if there are certain elements of the dataset
#       which lend themselves to visualizations especially well to be included in a manuscript for example.
EXAMPLE_INDICES: t.List[int] = []
# :param IMPORTANCE_CHANNELS:
#       The number of importance channels reflected in the ground truth importances loaded from the dataset.
#       For classification datasets we set this to the number of classes by default
IMPORTANCE_CHANNELS = NUM_CLASSES
# The string key of the graph dict representations to use to retrieve the ground truth node importances
# ! Note: A visual graph dataset may have multiple different ground truth explanation annotations with
# different numbers of channels!
NODE_IMPORTANCES_KEY: t.Optional[str] = 'node_importances_2'
# The string key of the graph dict representations to use to retrieve the ground truth edge importances
EDGE_IMPORTANCES_KEY: t.Optional[str] = 'edge_importances_2'

# :param USE_NODE_COORDINATES:
#       If this flag is true, the "node_coordinates" will be added to the other node attributes
USE_NODE_COORDINATES: bool = False
# :param USE_EDGE_ATTRIBUTES:
#       If this flag is False, instead of using the "edge_attributes" of the dataset,
#       constant edge weights of 1 are going to be used as a substitute
#       (some GNNs cannot handle edge attributes!)
USE_EDGE_ATTRIBUTES: bool = True
# :param USE_EDGE_LENGTHS:
#       If this flag is true, the "edge_lengths" will be added to the other edge attributes
USE_EDGE_LENGTHS: bool = False


# == MODEL PARAMETERS ==
# These parameters can be used to configure the model
MODEL_NAME: str = 'MOCK'

# == TRAINING PARAMETERS ==
# These parameters control the training process of the neural network.

# The number of independent training process repetitions to get a statistical measure of the performance
REPETITIONS: int = 1
# This optimizer will be used during training
OPTIMIZER_CB: t.Callable = lambda: ks.optimizers.Nadam(learning_rate=0.01)
BATCH_SIZE: int = 256
EPOCHS: int = 200
DEVICE: str = 'gpu:0'

# == EVALUATION PARAMETERS ==
# These parameters control the evaluation process, which includes the drawing of visualizations and plots

# After how many elements a log step is printed to the console
LOG_STEP_EVAL: int = 100
# This is the batch size that is used during the evaluation of the test set.
BATCH_SIZE_EVAL: int = 256

# == EXPERIMENT PARAMETERS ==
BASE_PATH = os.getcwd()
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):

    vgd_name = os.path.basename(VISUAL_GRAPH_DATASET_PATH)
    e.info(f'Starting to train GNNX model on vgd "{vgd_name}"')

    # -- DATASET LOADING --

    metadata_map, index_data_map = load_visual_graph_dataset(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )

    # index_data_map = {int(index): data for index, data in index_data_map.items()}
    dataset_indices = list(sorted(index_data_map.keys()))
    dataset_indices_set = set(dataset_indices)
    dataset_length = len(index_data_map)

    # As the first step in the processing pipeline we need to get a list of graph dicts, which can then
    # later be turned into the tensors needed for the model.
    dataset: t.List[tc.GraphDict] = [None for _ in dataset_indices]
    for index, data in index_data_map.items():
        g = data['metadata']['graph']

        if NODE_IMPORTANCES_KEY is not None:
            g['node_importances'] = g[NODE_IMPORTANCES_KEY]
        else:
            g['node_importances'] = np.zeros(shape=(len(g['node_indices']), IMPORTANCE_CHANNELS))

        if EDGE_IMPORTANCES_KEY is not None:
            g['edge_importances'] = g[EDGE_IMPORTANCES_KEY]
        else:
            g['edge_importances'] = np.zeros(shape=(len(g['edge_indices']), IMPORTANCE_CHANNELS))

        if USE_NODE_COORDINATES:
            g['node_attributes'] = np.array([g['node_attributes'][i].tolist() + g['node_coordinates'][i].tolist()
                                             for i in g['node_indices']])

        if USE_EDGE_LENGTHS:
            g['edge_attributes'] = np.array([g['edge_attributes'][i].tolist() + [g['edge_lengths'][i]]
                                             for i, _ in enumerate(g['edge_indices'])])

        if not USE_EDGE_ATTRIBUTES:
            g['edge_attributes'] = np.ones(shape=(len(g['edge_indices']), 1))

        dataset[index] = g

    e.info(f'loaded dataset with {len(dataset)} elements')

    # -- MODEL TRAINING --

    @e.hook('process_dataset', default=True)
    def process_dataset(_e, dataset, train_indices, test_indices):
        x_train, y_train, x_test, y_test = process_graph_dataset(
            dataset,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        return x_train, y_train, x_test, y_test


    for rep in range(REPETITIONS):
        e.info(f'REPETITION ({rep+1}/{REPETITIONS})')
        e['rep'] = rep

        with tf.device(DEVICE):

            if USE_DATASET_SPLIT is not None:
                e.info(f'creating train test split from dataset...')
                train_indices = [index for index, data in index_data_map.items()
                                 if USE_DATASET_SPLIT in data['metadata']['train_indices']]
                test_indices = [index for index, data in index_data_map.items()
                                if USE_DATASET_SPLIT in data['metadata']['test_indices']]

            else:
                e.info(f'creating random train test split...')
                train_indices = random.sample(dataset_indices, k=int(TRAIN_RATIO * dataset_length))
                train_indices_set = set(train_indices)
                test_indices_set =dataset_indices_set.difference(train_indices_set)
                test_indices = list(test_indices_set)

            test_indices_set = set(test_indices) | set(EXAMPLE_INDICES)
            test_indices = list(test_indices_set)
            e.info(f'using {len(train_indices)} training elements and {len(test_indices)} test elements')
            e[f'train_indices/{rep}'] = train_indices
            e[f'test_indices/{rep}'] = test_indices

            num_examples = min(NUM_EXAMPLES, len(test_indices)) - len(EXAMPLE_INDICES)
            example_indices = EXAMPLE_INDICES + list(random.sample(test_indices, k=num_examples))
            e.info(f'using {len(example_indices)} as examples')
            e[f'example_indices/{rep}'] = example_indices

            e.info('converting dataset into ragged tensors for training...')
            x_train, y_train, x_test, y_test = e.apply_hook(
                'process_dataset',
                dataset=dataset,
                train_indices=train_indices,
                test_indices=test_indices
            )

            e.info('creating the model...')
            # :hook create_model:
            #       This hook is supposed to return a fully compiled kgcnn model object. Note that this
            #       model does not necessarily have to return the node and edge importance explanation
            #       tensors immediately from the __call__ method. These tensors can also be created by a
            #       post hoc procedure which can be implemented in the "query_model" hook.
            model: ks.models.Model = e.apply_hook('create_model')

            e.info('fitting the model...')

            history: dict = e.apply_hook(
                'fit_model',
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            e[f'history/{rep}'] = history
            e[f'epochs/{rep}'] = list(range(EPOCHS))

            # -- EVALUATING THE MODEL --
            e.info('evaluating test set...')
            # :hook query_model:
            #       This hook takes gets the model object as the argument, as well as the x_test and y_test
            #       tensors. It has to return a tuple of three elements: The first element is supposed to
            #       be the tensor with the actual predictions, the second is the node importance explanation
            #       tensor and the last is the edge importance explanation tensor. How these various tensors
            #       are created (post-hoc explanations version self-explaining) is not restricted.
            out_pred, ni_pred, ei_pred = e.apply_hook(
                'query_model',
                model=model,
                x=x_test,
                y=y_test,
                include_importances=True,
            )
            out_pred = np.array(out_pred.numpy())
            classes_pred = [np.argmax(a) for a in out_pred]
            ni_pred = np.array(ni_pred.numpy())
            ei_pred = np.array(ei_pred.numpy())

            out_true = np.array(y_test[0])
            classes_true = [np.argmax(a) for a in out_true]
            ni_true = np.array(y_test[1].numpy())
            ei_true = np.array(y_test[2].numpy())

            # ~ metrics for the total test set
            e.info('calculating prediction metrics...')
            acc_value = accuracy_score(classes_true, classes_pred)
            auc_value = roc_auc_score(out_true, out_pred)
            e[f'acc/{rep}'] = acc_value
            e[f'auc/{rep}'] = auc_value
            e.info(f' * prediction metrics:'
                   f' - acc: {acc_value:.2f}'
                   f' - auc: {auc_value:.2f}')

            e.info('calculating fidelity...')
            fidelities = e.apply_hook(
                'calculate_fidelity',
                model=model,
                indices_true=test_indices,
                x_true=x_test,
                y_true=y_test,
                out_pred=out_pred,
                ni_pred=ni_pred,
                ei_pred=ei_pred,
            )
            if fidelities:
                for index, fidelity in zip(test_indices, fidelities):
                    e[f'fidelity/{rep}/{index}'] = fidelity

            # ~ calculating explanation metrics
            e.info('calculating explanation metrics...')

            pred_node_importances = []
            pred_edge_importances = []

            true_node_importances = []
            true_edge_importances = []

            for c, index in enumerate(test_indices):
                g = index_data_map[index]['metadata']['graph']
                ni = array_normalize(ni_pred[c])
                ei = array_normalize(ei_pred[c])

                e[f"out/true/{rep}/{index}"] = float(g['graph_labels'][0])
                e[f"out/pred/{rep}/{index}"] = float(out_pred[c][0])

                e[f'ni/pred/{rep}/{index}'] = ni
                e[f'ei/pred/{rep}/{index}'] = ei

                pred_node_importances += ni_pred[c].flatten().tolist()
                pred_edge_importances += ei_pred[c].flatten().tolist()

                true_node_importances += ni_true[c].flatten().tolist()
                true_edge_importances += ei_true[c].flatten().tolist()

                # ~ sparsity of the explanations
                node_sparsity = float(np.mean(binary_threshold(ni, 0.5)))
                edge_sparsity = float(np.mean(binary_threshold(ei, 0.5)))
                e[f'node_sparsity/{rep}/{index}'] = node_sparsity
                e[f'edge_sparsity/{rep}/{index}'] = edge_sparsity

            try:
                mean_fidelity = np.mean(list(e[f'fidelity/{rep}'].values()))
            except KeyError:
                mean_fidelity = 0

            mean_node_sparsity = np.mean(list(e[f'node_sparsity/{rep}'].values()))
            mean_edge_sparsity = np.mean(list(e[f'edge_sparsity/{rep}'].values()))

            # ~ explanation accuracy
            # In the case that there are no ground truth importances (in which case we have set them to
            # be all zeros) this will raise an exception. And in that case we will just return the worst
            # result.
            try:
                node_auc = roc_auc_score(true_node_importances, pred_node_importances)
                edge_auc = roc_auc_score(true_edge_importances, pred_edge_importances)
            except ValueError:
                node_auc = 0.5
                edge_auc = 0.5

            e[f'node_auc/{rep}'] = node_auc
            e[f'edge_auc/{rep}'] = edge_auc

            e.info(f' * explanation metrics: '
                   f' - mean fidelity: {mean_fidelity:.2f}'
                   f' - mean node sparsity: {mean_node_sparsity:.2f}'
                   f' - mean edge sparsity: {mean_edge_sparsity:.2f}'
                   f' - node auc: {node_auc:.2f}'
                   f' - edge auc: {edge_auc:.2f}')

            # -- VISUALIZATION OF RESULTS --

            # ~ visualizing examples of the graph
            e.info('drawing example visualizations...')
            # First of all we need to query the model using only the elements of the example set
            ni_example_pred = []
            ei_example_pred = []
            for index in example_indices:
                ni_example_pred.append(e[f'ni/pred/{rep}/{index}'])
                ei_example_pred.append(e[f'ei/pred/{rep}/{index}'])

            # Then we can use the already existing method "create_importances_pdf" to create a pdf file
            # which contains the visualizations of the explanations
            pdf_path = os.path.join(e.path, f'examples_{rep:02d}.pdf')
            graph_list = [index_data_map[i]['metadata']['graph'] for i in example_indices]
            image_path_list = [index_data_map[i]['image_path'] for i in example_indices]
            node_positions_list = [g['node_positions'] for g in graph_list]
            create_importances_pdf(
                graph_list=graph_list,
                image_path_list=image_path_list,
                node_positions_list=node_positions_list,
                importances_map={
                    'model': (ni_example_pred, ei_example_pred)
                },
                output_path=pdf_path,
                importance_channel_labels=CLASS_LABELS,
                labels_list=[index_data_map[i]['metadata']['name'] for i in example_indices]
            )

            e.status()


with Skippable(), e.analysis:
    # Creating latex code to display the results in a table
    e.info('rendering latex table...')
    column_names = [
        r'Target Name',
        r'$\text{Accuracy} \uparrow $',
        r'$\text{AUC} \uparrow $',
        r'$\text{Fidelity} \uparrow$',
        r'$\text{Node Sparsity} \downarrow$',
        r'$\text{Edge Sparsity} \downarrow$',
        r'$\text{Node AUC} \uparrow$',
        r'$\text{Edge AUC} \uparrow$',
    ]
    rows = []

    row = []

    row.append('-')
    row.append([e[f'acc/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'auc/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'fidelity/{rep}/{index}']
                for rep in range(REPETITIONS)
                for index in e[f'test_indices/{rep}']])
    row.append([e[f'node_sparsity/{rep}/{index}']
                for rep in range(REPETITIONS)
                for index in e[f'test_indices/{rep}']])
    row.append([e[f'edge_sparsity/{rep}/{index}']
                for rep in range(REPETITIONS)
                for index in e[f'test_indices/{rep}']])
    row.append([e[f'node_auc/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'edge_auc/{rep}'] for rep in range(REPETITIONS)])

    rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_mean,
        caption=f'Results of {REPETITIONS} repetition(s) of ' + r'\textbf{' + MODEL_NAME + '}'
    )
    e.commit_raw('table.tex', table)
    pdf_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=pdf_path)
    e.info('rendered latex table')