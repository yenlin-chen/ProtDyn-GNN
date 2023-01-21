#!/usr/bin/env python
# coding: utf-8

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from sys import path as systemPath
systemPath.append('../../')

from modules.models import multiGCN
from modules.control_suite import Experiment
from modules.datasets import TNM_8A_11001
from modules.visualization import Plotter

from os import path
import torch
from torch import nn
# import torchinfo


self_dir = path.dirname(path.realpath(__file__))

if __name__ == '__main__':

    ####################################################################
    # set up experiment
    ####################################################################

    # initialize dataset
    dataset = TNM_8A_11001(set_name='sim95-ha5k',
                           go_thres=25,
                           entry_type='monomer')
    print(f'Size of dataset: {len(dataset)}')

    # initialize model
    model = multiGCN(
        n_dims=2,
        dim_node_feat=21, dim_pers_feat=625, dim_out=dataset.n_GO_terms,
        dim_node_hidden=256,
        dim_pers_embedding=512, dim_graph_embedding=512,
        dropout_rate=0.1,
        n_graph_layers=5
    )

    # experiment setup
    exp = Experiment(
        model,
        name_suffix=f'{dataset.set_name}-{dataset.go_thres}',
        rand_seed=69,
        save_dir=self_dir
    )
    exp.set_dataset(dataset)
    exp.set_loss_fn(nn.BCEWithLogitsLoss, pos_weight=dataset.pos_weight)
    exp.set_optimizer(torch.optim.Adam, lr=0.000025)

    # plotter setup
    plt = Plotter(save_dir=self_dir)

    ####################################################################
    # train
    ####################################################################

    exp.train_split(n_epochs=500, train_valid_ratio=0.9, batch_size=64)

    ####################################################################
    # plot stuff
    ####################################################################

    # plot pr_curve for model from the last iteration
    plt.plot_pr(*exp.get_pr_curve(exp.valid_dataloader),
                filename_suffix='last')

    # plot pr_curve for model with best validation accuracy
    exp.load_params(path.join(exp.save_dir, 'lowest_loss-model.pkl'))
    plt.plot_pr(*exp.get_pr_curve(exp.valid_dataloader),
                filename_suffix='lowest_loss')

    # plot pr_curve for model with best validation accuracy
    exp.load_params(path.join(exp.save_dir, 'best_f1-model.pkl'))
    plt.plot_pr(*exp.get_pr_curve(exp.valid_dataloader),
                filename_suffix='best_f1')
