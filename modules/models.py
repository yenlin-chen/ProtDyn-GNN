#!/usr/bin/env python
# coding: utf-8

import json
import sys
from os import path

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

from torch_geometric.nn import (
    GCNConv,
    # GATConv,
    # SAGEConv,
    global_max_pool,
    LayerNorm
)

class SingleGCN(nn.Module):
    def __init__(self,
                 dim_node_feat, dim_pers_feat, n_classes,
                 dim_node_hidden,
                 dim_pers_embedding, dim_graph_embedding,
                 dropout_rate,
                 n_graph_layers,
                 dim_hidden_ls=None):
        '''Instantiate all components with trainable parameters'''

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        self.n_classes = n_classes

        super().__init__()

        # the convolutional layers
        self.conv_block = nn.ModuleList([])
        for i in range(n_graph_layers):
            dim_input = dim_node_feat if i == 0 else dim_node_hidden

            conv = GCNConv(dim_input, dim_node_hidden)
            # default arguments to GCNConv (and MessagePassing)
            # aggr='add', improved=False, add_self_loops=True
            self.conv_block.append(conv)

        # linear layers to process convolutional features
        self.graph_block = nn.Sequential(
            nn.Linear(n_graph_layers * dim_node_hidden, dim_graph_embedding),
            nn.LayerNorm(dim_graph_embedding),
            nn.Dropout(p=dropout_rate),
            nn.ReLU()
        )

        # linear layers to process the persistence diagram
        self.pi_block = nn.Sequential(
            nn.Linear(dim_pers_feat, dim_pers_embedding),
            nn.LayerNorm(dim_pers_embedding),
            nn.Dropout(p=dropout_rate),
            nn.ReLU()
        ) if dim_pers_embedding else None

        # final block to combine everything above
        if dim_hidden_ls is None:
            dim_hidden_ls = [n_classes]
        else:
            dim_hidden_ls.append(n_classes)

        fc_modules = []
        dim_hidden_in = dim_graph_embedding + dim_pers_embedding
        for dim_hidden_out in dim_hidden_ls:
            fc_modules.append(nn.Linear(dim_hidden_in, dim_hidden_out))
            fc_modules.append(nn.LayerNorm(dim_hidden_out))
            fc_modules.append(nn.Dropout(p=dropout_rate))
            dim_hidden_in = dim_hidden_out

        self.fc_block = nn.Sequential(*fc_modules)

    def forward(self, data):

        '''Make connects between the components to complete the model'''

        x = data.x

        # pipe features from each layer of convolution into the fc layer
        jk_connection = []
        for layer_idx, conv in enumerate(self.conv_block):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            jk_connection.append(x)

        jk_connection = torch.cat(jk_connection, dim=1)
        x = global_max_pool(jk_connection, data.batch)

        if self.pi_block: # graph & persistence embedding (concatenated)
            x = torch.cat((self.graph_block(x), self.pi_block(data.pi.float())),
                          dim=1)
        else: # graph embedding only
            x = self.graph_block(x)

        x = self.fc_block(x)

        return x

    def save_args(self, save_dir):

        '''Save details on the model for future reference'''

        self.all_args['class_name'] = type(self).__name__
        with open(path.join(save_dir, 'model-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__

class MultiGCN(nn.Module):
    def __init__(self, n_dims,
                 dim_node_feat, dim_pers_feat, n_classes,
                 dim_node_hidden,
                 dim_pers_embedding, dim_graph_embedding,
                 dropout_rate,
                 n_graph_layers,
                 dim_hidden_ls=None):
        '''Instantiate all components with trainable parameters'''

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        self.n_classes = n_classes
        self.n_graph_dims = n_dims

        super().__init__()

        # the convolutional layers
        self.conv_block_list = nn.ModuleList([])
        for dim_idx in range(n_dims):
            conv_block = nn.ModuleList([])
            for layer_idx in range(n_graph_layers):
                dim_input = dim_node_feat if layer_idx == 0 else dim_node_hidden
                conv = GCNConv(dim_input, dim_node_hidden)
                conv_block.append(conv)
            self.conv_block_list.append(conv_block)

        # linear layers for processing convoluted vectors
        self.graph_block = nn.Sequential(
            nn.Linear(n_dims * n_graph_layers * dim_node_hidden,
                      dim_graph_embedding),
            nn.LayerNorm(dim_graph_embedding),
            nn.Dropout(p=dropout_rate),
            nn.ReLU()
        )

        # linear layers to process the persistence diagram
        self.pi_block = nn.Sequential(
            nn.Linear(dim_pers_feat, dim_pers_embedding),
            nn.LayerNorm(dim_pers_embedding),
            nn.Dropout(p=dropout_rate),
            nn.ReLU()
        ) if dim_pers_embedding else None


        # final block to combine everything above
        fc_modules = []

        if dim_hidden_ls is None:
            dim_hidden_ls = [n_classes]
        else:
            dim_hidden_ls.append(n_classes)

        dim_hidden_in = dim_graph_embedding + dim_pers_embedding
        for dim_hidden_out in dim_hidden_ls:
            fc_modules.append(nn.Linear(dim_hidden_in, dim_hidden_out))
            fc_modules.append(nn.LayerNorm(dim_hidden_out))
            fc_modules.append(nn.Dropout(p=dropout_rate))
            dim_hidden_in = dim_hidden_out

        self.fc_block = nn.Sequential(*fc_modules)

    def forward(self, data):
        '''Make connects between the components to complete the model'''

        # pipe features from each layer of convolution into the fc layer
        jk_connection = []
        for dim_idx in range(self.n_graph_dims):
            x = data.x
            dim_slice = torch.argwhere(data.edge_type[:,dim_idx]==1).squeeze()
            dim_edge_index = data.edge_index[:, dim_slice]
            for layer_idx, conv in enumerate(self.conv_block_list[dim_idx]):
                x = conv(x, dim_edge_index)
                x = F.relu(x)
                jk_connection.append(x)

        jk_connection = torch.cat(jk_connection, dim=1)
        x = global_max_pool(jk_connection, data.batch)

        if self.pi_block: # graph & persistence embedding (concatenated)
            x = torch.cat((self.graph_block(x), self.pi_block(data.pi.float())),
                          dim=1)
        else: # graph embedding only
            x = self.graph_block(x)

        x = self.fc_block(x)

        return x

    def save_args(self, save_dir):
        with open(path.join(save_dir, 'model-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__

class mGCN(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
    def save_args(self, save_dir):
        with open(path.join(save_dir, 'model-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__
