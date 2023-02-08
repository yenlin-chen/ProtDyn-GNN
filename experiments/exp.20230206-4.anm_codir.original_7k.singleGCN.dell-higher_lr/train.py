#!/usr/bin/env python
# coding: utf-8

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from os import path
from sys import path as systemPath
systemPath.append(path.join('..', '..'))

from modules.models import SingleGCN
from modules.control_suite import Experiment
from modules.datasets import ANM_8A_11001_temporary
from modules.visualization import Plotter

import torch
from torch import nn

self_dir = path.dirname(path.realpath(__file__))

if __name__ == '__main__':

    n_epochs = 500
    batch_size = 128

    learning_rate = 0.00025

    plot_freq = 25 # default: 25

    ####################################################################
    # set up experiment
    ####################################################################

    # initialize dataset
    dataset = ANM_8A_11001_temporary(
        set_name='original_7k',
        go_thres=25,
        entry_type='monomer'
    )
    print(f'Size of dataset: {len(dataset)}')

    # initialize model
    model = SingleGCN(
        dim_node_feat=21, dim_pers_feat=625, n_classes=dataset.n_GO_terms,
        dim_node_hidden=256,
        dim_pers_embedding=512, dim_graph_embedding=512,
        dropout_rate=0.1,
        n_graph_layers=5,
        dim_hidden_ls=None
    )

    # experiment setup
    exp = Experiment(
        name_suffix=f'{dataset.set_name}-{dataset.go_thres}',
        rand_seed=69,
        save_dir=self_dir
    )
    exp.set_model(model)
    exp.set_dataset(dataset)
    exp.set_loss_fn(nn.BCEWithLogitsLoss, pos_weight=dataset.pos_weight)
    exp.set_optimizer(torch.optim.AdamW, lr=learning_rate)

    ####################################################################
    # train
    ####################################################################

    exp.train_split(
        n_epochs=n_epochs,
        train_valid_ratio=0.9,
        batch_size=batch_size,
        plot_freq=plot_freq
    )
