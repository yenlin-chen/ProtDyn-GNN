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

class singleGCN(nn.Module):
    def __init__(self,
                 dim_node_feat, dim_pers_feat, dim_out,
                 dim_node_hidden,
                 dim_pers_embedding, dim_graph_embedding,
                 dropout_rate,
                 n_graph_layers,
                 gradCAM=False):
        '''Instantiate all components with trainable parameters'''

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        self.dim_node_hidden = dim_node_hidden

        super().__init__()

        # the convolutional layers
        self.conv_block = nn.ModuleList([])
        for i in range(n_graph_layers):
            dim_input = dim_node_feat if i == 0 else dim_node_hidden

            conv = GCNConv(dim_input, dim_node_hidden)
            # default arguments to GCNConv (and MessagePassing)
            # aggr='add', improved=False, add_self_loops=True
            self.conv_block.append(conv)

        if gradCAM:
            pass

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

        # final layers to combine everything above
        dim_total_embedding = dim_graph_embedding + dim_pers_embedding
        self.fc_block = nn.Sequential(
            nn.Linear(dim_total_embedding, dim_out),
            nn.LayerNorm(dim_out),
            nn.Dropout(p=dropout_rate)
        )

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

class multiGCN(nn.Module):
    def __init__(self, n_dims,
                 dim_node_feat, dim_pers_feat, dim_out,
                 dim_node_hidden,
                 dim_pers_embedding, dim_graph_embedding,
                 dropout_rate,
                 n_graph_layers,
                 gradCAM=False):
        '''Instantiate all components with trainable parameters'''

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        self.n_dims = n_dims
        self.dim_node_hidden = dim_node_hidden

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

        # final layers to combine everything above
        dim_total_embedding = dim_graph_embedding + dim_pers_embedding
        self.fc_block = nn.Sequential(
            nn.Linear(dim_total_embedding, dim_out),
            nn.LayerNorm(dim_out),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, data):
        '''Make connects between the components to complete the model'''

        # pipe features from each layer of convolution into the fc layer
        jk_connection = []
        for dim_idx in range(self.n_dims):
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
