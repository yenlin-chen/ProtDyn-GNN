#!/usr/bin/env python
# coding: utf-8

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from sys import path as systemPath
systemPath.append('../../')

from preprocessing import (
    preprocessor as pp,
    utils
)

if __name__ == '__main__':

    for set_name in ['sim100-ha5k']:
        process = pp.Preprocessor(set_name=set_name,
                                  entry_type='monomer',
                                  go_thres=25,
                                  verbose=True)
        # id_mfgo = process.gen_labels(retry_download=True,
        #                              redownload=False,
        #                              verbose=None)
        process.preprocess(simplex=pp.df_simplex,
                           enm_type='tnm',
                           cutoff=8, n_modes=pp.df_n_modes,
                           retry_download=False,
                           rebuild_pi=False, rebuild_graph=False,
                           update_mfgo=True, verbose=None)
        process.preprocess(simplex=pp.df_simplex,
                           enm_type='anm',
                           cutoff=8, n_modes=pp.df_n_modes,
                           retry_download=False,
                           rebuild_pi=False, rebuild_graph=False,
                           update_mfgo=True, verbose=None)

    for set_name in ['original_7k']:
        process = pp.Preprocessor(set_name=set_name,
                                  entry_type='monomer',
                                  go_thres=25,
                                  verbose=True)
        # id_mfgo = process.gen_labels(retry_download=False,
        #                              redownload=False,
        #                              verbose=None)
        process.preprocess(simplex=pp.df_simplex,
                           enm_type='anm',
                           cutoff=8, n_modes=pp.df_n_modes,
                           retry_download=False,
                           rebuild_pi=False, rebuild_graph=False,
                           update_mfgo=True, verbose=None)
        process.preprocess(simplex=pp.df_simplex,
                           enm_type='tnm',
                           cutoff=8, n_modes=pp.df_n_modes,
                           retry_download=False,
                           rebuild_pi=False, rebuild_graph=False,
                           update_mfgo=True, verbose=None)
