#!/usr/bin/env python
# coding: utf-8

from sys import path as systemPath
systemPath.append('../../')

from preprocessing import (
    preprocessor as pp,
    utils
)

if __name__ == '__main__':

    process = pp.Preprocessor(set_name='test',
                              entry_type='monomer',
                              go_thres=2,
                              verbose=True,
                              coord_and_deform_coupling_not_implemented=True)
    id_mfgo = process.gen_labels(retry_download=False,
                                 redownload=False,
                                 verbose=None)
    process.preprocess(simplex=pp.df_simplex,
                       enm_type='anm',
                       cutoff=8, n_modes=pp.df_n_modes,
                       retry_download=False,
                       rebuild_pi=False, rebuild_graph=False,
                       update_mfgo=True, verbose=None)
