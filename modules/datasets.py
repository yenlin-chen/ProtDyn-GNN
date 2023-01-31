#!/usr/bin/env python
# coding: utf-8

from sys import path as systemPath
systemPath.append('..')

from preprocessing import (
    preprocessor as pp,
    utils,
    res1_dict, # for conversion from 1-letter symbol to residue indices
    res3_dict # for conversion from 3-letter symbol to residue indices
)

from os import path
import json
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.transforms import Compose
from tqdm import tqdm


module_dir = path.dirname(path.realpath(__file__))
pyg_cache_root = path.join(module_dir, 'pyg_cache')

# def rm_contact(data):
#     data.edge_type[:,0] = 0
#     return data
# def rm_codir(data):
#     data.edge_type[:,1] = 0
#     return data
# def rm_coord(data):
#     data.edge_type[:,2] = 0
#     return data
# def rm_deform(data):
#     data.edge_type[:,3] = 0
#     return data
# def rm_pi(data):
#     del data.pi
#     return data
# def rm_blank_edges(data):
#     keep_slice = torch.sum(data.edge_type, 1) > 0
#     # remove unwanted edges and edges info
#     data.edge_index = data.edge_index[:,keep_slice]
#     data.edge_type  = data.edge_type[keep_slice]
#     return data

# transform_list = np.array(
#     [rm_contact, rm_codir, rm_coord, rm_deform, rm_pi, rm_blank_edges],
#     dtype=np.object_
# )

class ProDAR_Dataset(pyg.data.Dataset):

    def __init__(self, set_name, go_thres, entry_type, enm_type,
                 cont, codir, coord, deform, pers,
                 cutoff, n_modes, simplex, transform=None):

        '''
        Base class for all datasets used in this project.
        '''

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        print('Initializing dataset...')

        if not any((cont, codir, coord, deform)):
            raise ValueError('At least one graph type must be turned on.')
        else:
            print('  Contact map is on.') if cont else None
            print('  Co-directionality coupling is on.') if codir else None
            print('  Coordination coupling is on.') if coord else None
            print('  Deformation coupling is on.') if codir else None
            print('  Persistence homology is on.') if pers else None

        self.set_name = set_name
        self.go_thres = go_thres
        self.entry_type = entry_type
        self.enm_type = enm_type
        if entry_type != 'monomer':
            raise NotImplementedError('Only monomers are supported at '
                                      'the moment')

        ################################################################
        # folder names containing data of the specified setup
        ################################################################
        if enm_type == 'anm':
            self.nma_setup = pp.anm_setup_folder.format(cutoff, n_modes)
        elif enm_type == 'tnm':
            self.nma_setup = pp.tnm_setup_folder.format(cutoff)
        else:
            raise NotImplementedError('Choose between anm and tnm for '
                                      'enm type')
        self.pi_setup = pp.pi_setup_folder.format(simplex)
        self.go_thres_setup = pp.go_thres_folder.format(self.go_thres)

        self.folder = f'{self.nma_setup}-{self.pi_setup}'

        self.raw_graph_dir = path.join(pp.df_graph_root, self.nma_setup)
        self.raw_pi_dir = path.join(pp.df_pi_root, self.pi_setup)

        self.labels_root = path.join(pp.df_labels_root, set_name)

        self.annotations_dir = path.join(self.labels_root, 'annotations')
        self.labels_dir = path.join(self.annotations_dir,
                                    f'{self.nma_setup}-{self.pi_setup}',
                                    self.go_thres_setup)

        ################################################################
        # save all dataset parameters
        ################################################################
        self.cont = cont
        self.codir = codir
        self.coord = coord
        self.deform = deform
        self.pers = pers

        self.cutoff = cutoff
        # self.gamma = gamma
        # self.corr_thres = corr_thres
        self.n_modes = n_modes

        self.simplex = simplex

        ################################################################
        # get list of IDs for this dataset
        ################################################################
        file = path.join(self.labels_dir, pp.df_mfgo_filename)

        # file exists if the preprocessor was executed on this dataset
        if path.exists(file):
            self.mfgo_file = file
            self.n_GO_terms = np.unique([e for v in self.mfgo_dict.values()
                                           for e in v]).size
        # run preprocessor if it was not executed before
        else:
            self.download()

        ################################################################
        # construct the transformation before data access
        ################################################################
        # binary_flags = np.array([cont, codir, coord, deform, pers],
        #                         dtype=np.bool_)
        # if binary_flags.all():
        #     transform = None
        # else:
        #     # dimension off = transformation on
        #     binary_flags = np.logical_not(binary_flags)
        #     # rm_blank_edges not required if all dimensions are on
        #     binary_flags = np.append(binary_flags, not (cont&codir&coord&deform))
        #     transform = Compose(transform_list[binary_flags])

        # def transform(data):
        #     # only keep the dimensions wanted
        #     data.edge_type = data.edge_type[:, binary_flags[:4]]
        #     # remove empty edges and edges info
        #     keep_slice = torch.sum(data.edge_type, 1) > 0
        #     data.edge_index = data.edge_index[:,keep_slice]
        #     data.edge_type  = data.edge_type[keep_slice]
        #     if not pers:
        #         del data.pi
        #     return data

        ################################################################
        # Call constuctor of parent class
        ################################################################
        super().__init__(self.raw_graph_dir, transform, None, None)
        print('Dataset Initialization Complete\n')

    @property
    def mfgo_dict(self):
        with open(self.mfgo_file, 'r') as f_in:
            mfgo_dict = json.load(f_in)
        return mfgo_dict

    @property
    def id_list(self):
        return [ID for ID in self.mfgo_dict]

    @property
    def pos_weight(self):
        mfgo_list = [e for v in self.mfgo_dict.values() for e in v]
        unique, count = np.unique(mfgo_list,
                                  return_counts=True)
        pos_weight = ( len(self.mfgo_dict)-count ) / count
        return torch.from_numpy(pos_weight)

    @property
    def raw_dir(self):
        return self.raw_graph_dir

    @property
    def processed_dir(self):
        return path.join(pyg_cache_root, self.folder)

    @property
    def raw_file_names(self):
        return [f'{ID}.json' for ID in self.id_list]

    @property
    def processed_file_names(self):
        return [f'{ID}.pt' for ID in self.id_list]

    def save_args(self, save_dir):

        self.all_args['class_name'] = type(self).__name__
        del self.all_args['transform']
        with open(path.join(save_dir, 'prodar_dataset-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

    def download(self):
        '''
        Run preprocessor processes to generate raw data.
        Uses existing label file if exists.
        '''

        template_dir = path.join(self.annotations_dir, 'template')

        process = pp.Preprocessor(set_name=self.set_name,
                                  go_thres=self.go_thres,
                                  entry_type=self.entry_type)

        if not path.exists(path.join(template_dir, pp.df_mfgo_filename)):
            process.gen_labels(retry_download=False,
                               redownload=False,
                               verbose=True)

        process.preprocess(simplex=self.simplex,
                           enm_type=self.enm_type,
                           cutoff=self.cutoff,
                           # gamma=self.gamma,
                           # corr_thres=self.corr_thres,
                           n_modes=self.n_modes,
                           retry_download=False,
                           rebuild_pi=False,
                           rebuild_graph=False,
                           update_mfgo=True,
                           verbose=True)
        self.mfgo_file = path.join(self.labels_dir, pp.df_mfgo_filename)
        self.n_GO_terms = np.unique([e for v in self.mfgo_dict.values()
                                       for e in v]).size

    def process(self):
        for idx, ID in enumerate(tqdm(self.id_list,
                                      desc='  Processing data (PyG)',
                                      ascii=True, dynamic_ncols=True)):
            try:
                with open(path.join(self.raw_graph_dir, ID+'.json'), 'r') as fin:
                    js_graph = json.load(fin)
            except ValueError as err:
                tqdm.write(f'Error reading {ID}.json in {self.raw_graph_dir}')
                raise err

            nx_graph = nx.readwrite.json_graph.node_link_graph(js_graph)
            data = pyg.utils.from_networkx(nx_graph)

            ############################################################
            # residue type
            ############################################################

            n_unique_residues = np.unique(list(res3_dict.values()))
            x = np.zeros((len(data.resname), len(n_unique_residues)),
                         dtype=np.int_)

            if self.enm_type == 'tnm':
                for j, res_code1 in enumerate(data.resname):
                    if res_code1 not in res1_dict:
                        res_code1 = 0
                    x[j, res1_dict[res_code1]] = 1
            elif self.enm_type == 'anm':
                for j, res_code3 in enumerate(data.resname):
                    if res_code3 not in res3_dict:
                        res_code3 = 'XAA'
                    x[j, res3_dict[res_code3]] = 1
            else:
                raise NotImplementedError()

            data.x = torch.from_numpy(x)

            pi = np.load(path.join(self.raw_pi_dir, ID+'.npy'))
            data.pi = torch.from_numpy(pi).float()

            ############################################################
            # labels
            ############################################################

            data.ID = ID

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects
            del data.resname

            processed_filename = ID+'.pt'
            torch.save(data, path.join(self.processed_dir, processed_filename))

    def len(self):
        return len(self.mfgo_dict)

    def get(self, idx):
        return torch.load(path.join(self.processed_dir,
                                    self.processed_file_names[idx]))

def transform_ANM_8A_10001_temporary(data):
    # only keep the dimensions wanted
    data.edge_type = data.edge_type[:, [True, False]]
    # remove empty edges and edges info
    keep_slice = torch.sum(data.edge_type, 1) > 0
    data.edge_index = data.edge_index[:,keep_slice]
    data.edge_type  = data.edge_type[keep_slice]
    return data

class ANM_8A_10001_temporary(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, False, True, True, True

        enm_type = 'anm'
        cutoff = 8

        transform = transform_ANM_8A_10001_temporary

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

class ANM_8A_11001_temporary(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, True, True, True, True

        enm_type = 'anm'
        cutoff = 8

        transform = None

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

def transform_TNM_8A_10001(data):
    # only keep the dimensions wanted
    data.edge_type = data.edge_type[:, [True, False, False, False]]
    # remove empty edges and edges info
    keep_slice = torch.sum(data.edge_type, 1) > 0
    data.edge_index = data.edge_index[:,keep_slice]
    data.edge_type  = data.edge_type[keep_slice]
    return data

class TNM_8A_10001(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, False, False, False, True

        enm_type = 'tnm'
        cutoff = 8

        transform = transform_TNM_8A_10001

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

class TNM_8A_all(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, True, True, True, True

        enm_type = 'tnm'
        cutoff = 8

        transform = None

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

def transform_TNM_8A_11001(data):
    # only keep the dimensions wanted
    data.edge_type = data.edge_type[:, [True, True, False, False]]
    # remove empty edges and edges info
    keep_slice = torch.sum(data.edge_type, 1) > 0
    data.edge_index = data.edge_index[:,keep_slice]
    data.edge_type  = data.edge_type[keep_slice]
    return data

class TNM_8A_11001(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, True, False, False, True

        enm_type = 'tnm'
        cutoff = 8

        transform = transform_TNM_8A_11001

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

def transform_TNM_8A_10101(data):
    # only keep the dimensions wanted
    data.edge_type = data.edge_type[:, [True, False, True, False]]
    # remove empty edges and edges info
    keep_slice = torch.sum(data.edge_type, 1) > 0
    data.edge_index = data.edge_index[:,keep_slice]
    data.edge_type  = data.edge_type[keep_slice]
    return data

class TNM_8A_10101(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, False, True, False, True

        enm_type = 'tnm'
        cutoff = 8

        transform = transform_TNM_8A_10101

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

def transform_TNM_8A_10011(data):
    # only keep the dimensions wanted
    data.edge_type = data.edge_type[:, [True, False, False, True]]
    # remove empty edges and edges info
    keep_slice = torch.sum(data.edge_type, 1) > 0
    data.edge_index = data.edge_index[:,keep_slice]
    data.edge_type  = data.edge_type[keep_slice]
    return data

class TNM_8A_10011(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, False, False, True, True

        enm_type = 'tnm'
        cutoff = 8

        transform = transform_TNM_8A_10011

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

def transform_TNM_8A_11101(data):
    # only keep the dimensions wanted
    data.edge_type = data.edge_type[:, [True, True, True, False]]
    # remove empty edges and edges info
    keep_slice = torch.sum(data.edge_type, 1) > 0
    data.edge_index = data.edge_index[:,keep_slice]
    data.edge_type  = data.edge_type[keep_slice]
    return data

class TNM_8A_11101(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, codir, coord, deform, pers = True, True, True, False, True

        enm_type = 'tnm'
        cutoff = 8

        transform = transform_TNM_8A_11101

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type, enm_type=enm_type,
                         cutoff=cutoff,
                         cont=cont, codir=codir, coord=coord,
                         deform=deform, pers=pers,
                         # gamma=pp.df_gamma, corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex,
                         transform=transform)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def save_args(self, save_dir):
        return super().save_args(save_dir)

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)
