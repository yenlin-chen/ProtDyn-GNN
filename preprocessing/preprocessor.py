#!/usr/bin/env python
# coding: utf-8

from . import utils

import json
import requests
from os import (
    path,
    makedirs,
    chdir,
    getcwd,
    cpu_count,
    remove as remove_file
)
from tqdm import tqdm
from datetime import datetime
import numpy as np
import prody
import signal
from math import sqrt
import gudhi as gd
import gudhi.representations
import networkx as nx
import subprocess
import gzip
from shutil import copyfileobj

module_dir = path.dirname(path.realpath(__file__))

########################################################################
# GLOBAL VARIABLES (ALSO USED BY SCIPTS THAT IMPORT THIS MODULE)
########################################################################
# preprocessed data goes here
df_graph_root = path.join(module_dir, 'data', 'graphs')
df_pi_root = path.join(module_dir, 'data', 'persistence_images')
# df_rcsb_template = path.join(module_dir, 'rcsb-payload-template')

# stats on datasets can be found here
df_labels_root = path.join(module_dir, 'labels')

# save download time by saving a copy of downloaded raw data
cache_dir = path.join(module_dir, 'cache')
mfgo_cache_dir = path.join(cache_dir, 'mfgo')
cif_cache_dir = path.join(cache_dir, 'cif')
pdb_cache_dir = path.join(cache_dir, 'pdb')
tnm_cache_dir = path.join(cache_dir, 'tnm')
entity_cache_dir = path.join(cache_dir, 'rcsb-api', 'polymer_entity')
instance_cache_dir = path.join(cache_dir, 'rcsb-api',
                               'polymer_entity_instance')

# df_payload_filename = 'payload.json'
# df_payload_template = 'payload-template.json'
df_id_list_filename = 'id_list-asym_id.txt'
df_mfgo_cnt_filename = 'mfgo-count.txt'
df_mfgo_filename = 'id-mfgo.json'
df_noMFGO_filename = 'id-without_MFGO.txt'

df_chain_filename = 'chain-all.txt'
df_failed_chain_filename = 'chain-failed.txt'
df_pdb_filename = 'pdb-all.txt'
df_failed_pdb_filename = 'pdb-failed.txt'

tnm_setup_filename = '{}{}_MIN{:.1f}_ALL_PHIPSIPSI'

tnm_setup_folder = 'tnm-cutoff_{}A-codir_only'
anm_setup_folder = 'anm-cutoff_{}A-nModes_{}-codir_only'
# md_setup_folder = 'md'
pi_setup_folder = 'simplex_{}'
go_thres_folder = 'GOthres_{}'

url_entity = 'https://data.rcsb.org/rest/v1/core/polymer_entity'
url_instance = 'https://data.rcsb.org/rest/v1/core/polymer_entity_instance'

# makedirs(df_graph_root, exist_ok=True)
# makedirs(df_pi_root, exist_ok=True)

df_GET_timeout = 5

# default parameters
df_atomselect = 'calpha'

df_simplex = 'alpha'
df_pi_range = [0, 50, 0, 50*sqrt(2)/2]
df_pi_size = [25, 25]

df_cutoff = 8
df_gamma = 1
df_codir_thres = 0.5
df_n_modes = 20

########################################################################
# GLOBAL VARIABLES (ALSO USED BY SCIPTS THAT IMPORT THIS MODULE)
########################################################################
download_exe = path.join(module_dir, 'batch_download.sh')
tnm_exe = path.join(module_dir, 'tnm')

def format_pdb_chain_id(pdb_id, chain_id):
    return f'{pdb_id.upper()}-{chain_id}'

def format_pdb_entity_id(pdb_id, entity_id):
    return f'{pdb_id.upper()}_{entity_id}'

class Preprocessor():

    def __init__(self, set_name, entry_type, go_thres=25, verbose=True):

        '''
        Directory manager for data preprocessing. Also provides the
        functions required to build ProDAR datasets.

        Caches and logs are maintained to save significant time on
        subsequent runs.
        '''

        print('\nInstantiating preprocessor...', end='')

        # name of protein selection, e.g. original_7k
        self.set_name = set_name
        self.entry_type = entry_type # 'monomer' or 'complex'
        if self.entry_type != 'monomer':
            raise NotImplementedError(
                'Only monomers are supported at the moment'
            )
        self.go_thres = go_thres

        ################################################################
        # save and cache location
        ################################################################
        self.labels_root = path.join(df_labels_root, set_name)
        self.annotations_dir = path.join(self.labels_root, 'annotations')
        self.target_dir = path.join(self.labels_root, 'target')
        self.rcsb_dir = path.join(self.labels_root, 'rcsb-search-api')

        # for saving annotations prior to data processing (entries that
        # failed to process will be removed from this set to obtain the
        # final annotation for the dataset)
        self.template_dir = path.join(self.annotations_dir, 'template')
        # go_thres_folder.format(go_thres)) # DEFUNCT

        # keep a list of failed entries to skip on later executions
        self.log_dir = path.join(module_dir, 'log')

        ################################################################
        # filename of log read by all datasets (across all executions)
        ################################################################
        self.mfgo_log = path.join(self.log_dir,
                                  'pdb_id-failed_to_label-all.log')
        self.download_parse_log = path.join(
            self.log_dir,
            f'pdb_id-failed_to_download_parse-all.log'
        )

        ################################################################
        # filename for dataset-specific log
        ################################################################
        self.local_mfgo_logname = 'id-failed_to_label.log'
        self.local_process_logname = 'id-failed_to_process.log'
        # self.entity2chain_logname = 'id-failed_to_convert_entity2chain.log'

        self.verbose = verbose

        ################################################################
        # create directories
        ################################################################
        makedirs(self.labels_root, exist_ok=True)
        makedirs(self.template_dir, exist_ok=True)
        makedirs(mfgo_cache_dir, exist_ok=True)
        makedirs(cif_cache_dir, exist_ok=True)
        makedirs(pdb_cache_dir, exist_ok=True)
        makedirs(tnm_cache_dir, exist_ok=True)
        makedirs(entity_cache_dir, exist_ok=True)
        makedirs(instance_cache_dir, exist_ok=True)
        makedirs(self.log_dir, exist_ok=True)

        ################################################################
        # set up ProDy
        ################################################################
        prody.confProDy(verbosity='none')
        prody.pathPDBFolder(folder=pdb_cache_dir, divided=False)

        print('Done', end='\n\n')

        self.go_url = 'https://www.ebi.ac.uk/pdbe/api/mappings/go/'

    def _get_instance_info(self, pdb_id, chain_id_rcsb, redownload,
                           verbose=None):

        '''
        Get various IDs of the instance, e.g. entity_id, auth_asym_id
        '''

        instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)
        instance_cache = path.join(instance_cache_dir, f'{instance_id}.json')

        redownload_msg = False
        if redownload and path.exists(instance_cache):
            remove_file(instance_cache)
            redownload_msg = True

        # access instance data from RCSB
        if path.exists(instance_cache): # check if local copy exists
            msg = 'Instance data loaded from disk'
            with open(instance_cache, 'r') as f_in:
                data = json.load(f_in)
        else:
            url_endpoint = f'{url_instance}/{pdb_id}/{chain_id_rcsb}'
            try:
                retrieved = requests.get(url_endpoint, timeout=df_GET_timeout)
            except request.Timeout:
                msg = 'Timeout while GET request for instance data'
                utils.vprint(verbose, msg)
                return None, msg

            if retrieved.status_code != 200:
                msg = (f'GET request for instance data failed with code '
                       f'{retrieved.status_code}')
                utils.vprint(verbose, msg)
                return None, msg

            msg = ('Instance data re-retrieved from RCSB' if redownload_msg
                   else 'Instance data retrieved from RCSB')
            data = retrieved.json()
            # save a local copy
            with open(instance_cache, 'w+') as f_out:
                json.dump(data, f_out,
                          indent=4, separators=(',', ': '), sort_keys=False)

        return data['rcsb_polymer_entity_instance_container_identifiers'], msg

    def chain_id_rcsb2auth(self, pdb_id, chain_id_rcsb, redownload,
                           verbose=None):

        ids, msg = self._get_instance_info(pdb_id, chain_id_rcsb, redownload,
                                           verbose)
        if ids is None:
            return None, msg

        chain_id_auth = ids['auth_asym_id']
        return chain_id_auth, msg

    def chain_id_rcsb2entity_id(self, pdb_id, chain_id_rcsb,
                                redownload, verbose=None):

        ids, msg = self._get_instance_info(pdb_id, chain_id_rcsb, redownload,
                                           verbose)
        if ids is None:
            return None, msg

        entity_id = ids['entity_id']
        return entity_id, msg

    def _get_mfgo(self, pdb_id, chain_id_rcsb, redownload=False, verbose=None):

        '''
        Retrieves the MF-GO annotation for the given PDB entry.
        Returns the annotations as a dictionary, along with a string
        explaining the reason of success or failure of the process.
        '''

        if chain_id_rcsb is None:
            raise NotImplementedError(
                'Finding MF-GO terms for multimers (RCSB entries) are '
                'currently unavailable'
            )

        instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose, f'Retrieving MF-GO for \'{instance_id}\'...',
                     end='', flush=True)

        entity_id, _ = self.chain_id_rcsb2entity_id(pdb_id, chain_id_rcsb,
                                                    redownload, verbose)
        entity_cache = path.join(entity_cache_dir,
                             f'{format_pdb_entity_id(pdb_id, entity_id)}.json')

        redownload_msg = False
        if redownload and path.exists(entity_cache):
            remove_file(entity_cache)
            redownload_msg = True

        # access entity data from RCSB
        if path.exists(entity_cache): # check if local copy exists
            msg = 'Entity data found on disk'
            with open(entity_cache, 'r') as f_in:
                data = json.load(f_in)
        else:# check if local copy exists
            url_endpoint = f'{url_entity}/{pdb_id}/{entity_id}'
            try:
                retrieved = requests.get(url_endpoint, timeout=df_GET_timeout)
            except request.Timeout:
                msg = 'Timeout while GET request for entity data'
                utils.vprint(verbose, msg)
                return None, msg

            if retrieved.status_code != 200:
                msg = (f'GET request for entity data failed with code '
                       f'{retrieved.status_code}')
                utils.vprint(verbose, msg)
                return None, msg

            msg = ('Entity data re-retrieved from RCSB' if redownload_msg
                   else 'Entity data retrieved from RCSB')
            data = retrieved.json()
            with open(entity_cache, 'w+') as f_out:
                json.dump(data, f_out,
                          indent=4, separators=(',', ': '), sort_keys=False)

        # read MF-GO annotations
        # mfgo_endpoints = []
        mfgo_all = []
        if 'rcsb_polymer_entity_annotation' in data:
            for anno in data['rcsb_polymer_entity_annotation']:
                mfgo_id = anno['annotation_id']

                # look for GO tags
                if not mfgo_id[:2] == 'GO':
                    continue
                # look for the molecular_function tag (GO:0003674)
                mfgo_lineage = [e['id'] for e in anno['annotation_lineage']]
                if not 'GO:0003674' in mfgo_lineage:
                    continue

                # mfgo_endpoints.append(mfgo_id)
                mfgo_all += mfgo_lineage
        mfgo_all = np.unique(mfgo_all)

        # msg = 'Re-downloaded' if redownload_msg else 'Downloaded'
        utils.vprint(verbose, msg)
        return mfgo_all, msg

    # def get_similarity_dataset(self, sim_cutoff, lt_heavy_atoms=5000):
    #     pass

    def gen_labels(self, retry_download=False, redownload=False, verbose=None):

        '''
        Retrieves MF-GO terms using the RCSB Search API and generates the
        labels for all IDs base on the list of GO entries (the dimension
        of the label is equal to the length of the list)
        '''

        id_list_file = path.join(self.target_dir, df_id_list_filename)
        print(f'Processing {id_list_file}')
        id_list = np.loadtxt(id_list_file, dtype=np.str_)
        if id_list.size == 1:
            id_list = np.array([id_list])

        print('Generating labels...')
        verbose = self.verbose if verbose is None else verbose

        # holder for PBDs/chains that are successfully preprocessed
        processed_ids = []
        failed_ids = []

        # dataset-specific log
        dataset_log = path.join(self.template_dir, self.local_mfgo_logname)
        open(dataset_log, 'w+').close() # clear file content

        # backup and clear logfile
        if retry_download and path.exists(self.mfgo_log):
            utils.backup_file(self.mfgo_log)
            # the list of IDs to skip will be empty
            utils.rm_log_entries(self.mfgo_log, id_list)

        # get list of PDBs/chains that should be skipped
        log_content, logged_ids = utils.read_logs(self.mfgo_log)
        # logged_ids = [ID for ID in logged_ids] # defunct
        utils.vprint(verbose,
                     f' -> {len(logged_ids)} PDB entries found in log, '
                     f'{len(set(logged_ids) & set(id_list))} to be skipped')

        ################################################################
        # retrieve MF-GO annotations
        ################################################################
        id_mfgo = {}
        for ID in tqdm(id_list, unit=' entries',
                       desc='Retrieving MF-GO',
                       ascii=True, dynamic_ncols=True):

            pdb_id = ID[:4]
            chain_id_rcsb = ID[5:] if len(ID)>5 else None
            instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

            ############################################################
            # skip if the PDB/chain failed to download in a previous run
            ############################################################
            if instance_id in logged_ids:
                # copy entry to dataset-specific log
                idx = logged_ids.index(instance_id)
                utils.append_to_file(log_content[idx], dataset_log)
                failed_ids.append(instance_id)
                tqdm.write(f'  Skipping \'{instance_id}\'')
                continue

            # if the PDB entry was not skipped
            tqdm.write(f'  Processing \'{instance_id}\'...')

            ############################################################
            # try to download MF-GO
            ############################################################
            tqdm.write('    Fetching MF-GO...')
            mfgo, msg = self._get_mfgo(pdb_id, chain_id_rcsb,
                                            redownload=redownload,
                                            verbose=False)
            tqdm.write(f'        {msg}')
            if mfgo is None:
                utils.append_to_file(f'{instance_id} -> MF-GO: {msg}',
                                     dataset_log)
                utils.append_to_file(f'{instance_id} -> MF-GO: {msg}',
                                     self.mfgo_log)
                failed_ids.append(instance_id)
                continue
            else:
                processed_ids.append(instance_id)
                id_mfgo[instance_id] = mfgo

        # get list of unique items
        mfgo_all = np.concatenate([e for e in id_mfgo.values()])
        mfgo_unique, mfgo_cnt = np.unique(mfgo_all, return_counts=True)

        # save info to drive
        np.savetxt(path.join(self.template_dir, df_mfgo_cnt_filename),
                   np.column_stack((mfgo_unique, mfgo_cnt)), fmt='%s %s')

        ################################################################
        # generate labels for dataset
        ################################################################
        for instance_id in tqdm(processed_ids, unit=' entries',
                                desc='Generating labels',
                                ascii=True, dynamic_ncols=True):

            labels = []
            for code in id_mfgo[instance_id]:
                # get index of the code
                loc = np.argwhere(mfgo_unique==code)
                # if there is no match (if num < self.go_thres)
                if loc.size == 0:
                    continue
                # append index to list if there is a match
                else:
                    labels.append(int(loc[0,0]))

            id_mfgo[instance_id] = labels

        # write labels to drive
        mfgo_file = path.join(self.template_dir, df_mfgo_filename)
        with open(mfgo_file, 'w+') as f_out:
            json.dump(id_mfgo, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        return id_mfgo

    def _get_struct(self, pdb_id, chain_id_rcsb, verbose=None):

        '''
        Retrieves protein structure through ProDy. Returns the structure
        as a ProDy 'atoms' type, the correct pdbID-chainID (uppercase,
        lowercase, etc.), and a string explaining the reason of success
        or failure of the process.

        Caution: use asym_id for chain ID (ProDy uses auth_asym_id)
        '''

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose,
                     f'Retrieving protein structure for '
                     f'\'{pdb_id}-{chain_id_rcsb}\'...',
                     end='', flush=True)

        # switch working directory to prevent ProDy from polluting
        # current working dir with .cif files
        cwd = getcwd()
        chdir(cif_cache_dir)

        # return data for all chains in PDB entry if no chains were
        # specified
        if chain_id_rcsb is None:
            raise NotImplementedError('Only supports monomers (chains), '
                                      'please specify chain ID.')
            atoms = prody.parsePDB(pdb_id, subset=df_atomselect)
            chdir(cwd) # switch working directory back ASAP

            # if parse was successful
            if atoms is not None:
                msg = 'Structure downloaded/parsed'
                utils.vprint(verbose, msg)
                return atoms, msg
            else: # parse failed
                msg = 'Cannot download PDB structure'
                utils.vprint(verbose, msg)
                return None, msg

        # if chain ID was specified
        else:
            chain_id_auth, _ = self.chain_id_rcsb2auth(pdb_id, chain_id_rcsb,
                                                       redownload=False,
                                                       verbose=False)

            try:
                atoms = prody.parsePDB(pdb_id,
                                       subset=df_atomselect,
                                       chain=chain_id_auth)
            except UnicodeDecodeError as errMsg:
                chdir(cwd)
                utils.vprint(verbose, errMsg)
                return None, errMsg

            # try to parse chain if structure is retrieved
            if atoms is not None:
                chdir(cwd)

                msg = 'Structure downloaded/parsed'
                utils.vprint(verbose, 'Done')
                return atoms, msg

                # else:
                #     msg = (f'ProDy cannot resolve chain ID (auth_asym_id) '
                #            f'{chain_id_auth}')
                #     utils.vprint(verbose, msg)
                #     return None, msg

            else: # ProDy cannot download structure
                chdir(cwd)
                msg = 'ProDy cannot download structure'
                utils.vprint(verbose, msg)
                return None, msg

    def _get_PI(self, pdb_id, chain_id_rcsb, save_dir, simplex,
                img_range=df_pi_range, img_size=df_pi_size,
                coords=None, rebuild_existing=False, verbose=None):

        '''
        Returns the persistence image for the specified pdb entry or
        chain.

        The function checks if the persistence image for the specified
        ID is saved on disk, and returns the data if it is found.
        Otherwise, the structure for the ID will be retrieved by
        _get_struct() if the coords is not give, and the newly computed
        product is saved on disk.
        '''

        instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose,
                     f'Retrieving persistence image for \'{instance_id}\'...',
                     end='', flush=True)

        pi_file = path.join(save_dir, f'{instance_id}.npy')

        rebuild_msg = False
        if rebuild_existing and path.exsits(pi_file):
            remove_file(pi_file)
            # change the output for rebuilt entries
            rebuild_msg = True

        if path.exists(pi_file):
            msg = 'Data found on disk'
            pers_img = np.load(pi_file)
            utils.vprint(verbose, msg)
            return pers_img, msg

        # try computing if file not found on disk
        if not coords: # use coords if given to save computation
            atoms, msg = self._get_struct(pdb_id, chain_id_rcsb, verbose=False)
            if atoms is None:
                utils.vprint(verbose, msg)
                return None, msg
            else:
                coords = atoms.getCoords().tolist()

        # simplicial  complex
        if simplex == 'alpha':
            scx = gd.AlphaComplex(points=coords).create_simplex_tree()
        elif simplex == 'rips':
            distMtx = sp.spatial.distance_matrix(coords, coords, p=2)
            scx = gd.RipsComplex(distance_matrix=distMtx).create_simplex_tree()

        # persistence image
        pi = gd.representations.PersistenceImage(
            bandwidth=1,
            weight=lambda x: max(0, x[1]*x[1]),
            im_range=img_range,
            resolution=img_size
        )

        scx.persistence()

        pInterval_d1 = scx.persistence_intervals_in_dimension(1)
        pInterval_d2 = scx.persistence_intervals_in_dimension(2)

        if pInterval_d1.size!=0 and pInterval_d2.size!=0:
            pers_img = pi.fit_transform([np.vstack((pInterval_d1,
                                                    pInterval_d2))])
        elif pInterval_d1.size!=0 and pInterval_d2.size==0:
            pers_img = pi.fit_transform([pInterval_d1])
        elif pInterval_d1.size==0 and pInterval_d2.size!=0:
            pers_img = pi.fit_transform([pInterval_d2])
        else:
            # discard PDB entry if size is 0 in both dimensions
            msg = 'Persistence interval in both dimensions are 0'
            utils.vprint(verbose, msg)
            return None, msg

        # if computation is successful
        msg = 'Rebuilt' if rebuild_msg else 'Computed'
        np.save(pi_file, pers_img)
        utils.vprint(verbose, msg)
        return pers_img, msg

    def comp_freqs(self, pdb_id, chain_id_rcsb, atoms=None,
                   cutoff=df_cutoff, # gamma=df_gamma,
                   n_modes=df_n_modes, nCPUs=cpu_count(), verbose=None):

        '''
        Computes and returns first few modes with ProDy for the
        specified PDB entry or chain, depending on whether a chain ID is
        specified in the argument 'ID'.
        '''

        instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose,
                     f'Computing modes for \'{instance_id}\'...',
                     end='', flush=True)

        if not atoms: # use coords if given to save computation
            atoms, msg = self._get_struct(pdb_id, chain_id_rcsb, verbose=False)
            if atoms is None:
                utils.vprint(verbose, msg)
                return None, None, msg

        anm = prody.ANM(name=instance_id)
        anm.buildHessian(atoms, cutoff=cutoff, gamma=df_gamma,
                         norm=True, n_cpu=nCPUs)

        # modal analysis
        try:
            anm.calcModes(n_modes=n_modes, zeros=False, turbo=True)
        except Exception as err:
            # discard PDB entry if normal mode analysis fails
            msg = 'Unable to compute modes.'
            utils.vprint(verbose, msg)
            return None, None, msg

        freqs = [sqrt(mode.getEigval()) for mode in anm]

        return anm, freqs, 'Computed'

    def _get_tnm_graph(self,  pdb_id, chain_id_rcsb, nma_dir,
                       cutoff,
                       nCPUs, rebuild_existing=False,
                       verbose=None):
        '''
        Returns the contact map, and 3 dynamical couplings from the
        results of NMA using TNM.

        Checks if the graph for the specified ID is already saved on
        disk, and returns the data if found. If not found, structure for
        the PDB entry or chain will be downloaded using _get_struct().
        Computation is delegated to the "official" TNM software.
        '''

        instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose,
                     f'Retrieving graphs (TNM) for \'{instance_id}\'...',
                     end='', flush=True)

        # define save directory for graphs
        graph_file = path.join(nma_dir, f'{instance_id}.json')

        rebuild_msg = False
        if rebuild_existing and path.exists(graph_file):
            remove_file(graph_file)
            # change output message if entry is rebuilt
            rebuild_msg = True

        # returns data found on disk if rebuild is not required
        if path.exists(graph_file):
            msg = 'Data found on disk'
            with open(graph_file, 'r') as f_in:
                graph_dict = json.load(f_in)
            self.all_resnames += [n['resname'] for n in graph_dict['nodes']]
            utils.vprint(verbose, msg)
            return graph_dict, msg

        if chain_id_rcsb is None:
            raise NotImplementedError('Only supports monomers (chains), '
                                      'please specify chain ID.')
            # gotta find all chains within the bioassembly
        else:
            chain_id_auth, _ = self.chain_id_rcsb2auth(pdb_id, chain_id_rcsb,
                                                       redownload=False,
                                                       verbose=False)

        utils.vprint(verbose, 'TNM modes...', flush=True)
        cache_dir = path.join(tnm_cache_dir, instance_id)
        makedirs(cache_dir, exist_ok=True)
        prefix = tnm_setup_filename.format(pdb_id, chain_id_rcsb, cutoff)

        script_file = path.join(cache_dir, f'{prefix}.in')
        tnm_log_file = path.join(cache_dir, f'{prefix}.log')

        # check if mapping (naming) file exists
        mapping_file = path.join(cache_dir, f'{prefix}.names.dat')
        file_existence = [path.exists(mapping_file)]
        # check if contact map file exists
        cont_file = path.join(cache_dir, f'{prefix}_Cont_Mat.txt')
        file_existence.append(path.exists(cont_file))
        # check if coupling files exist
        for dc in ['directionality', 'coordination', 'deformation']:
            file = path.join(cache_dir, f'{prefix}.{dc}_coupling.dat')
            file_existence.append(path.exists(file))

        ################################################################
        # obtain TNM results
        ################################################################
        # if TNM results do not already exist
        if not all(file_existence):
            # decompress the .pdb.gz file downloaded by ProDy
            compressed_file = path.join(pdb_cache_dir,
                                        f'{pdb_id.lower()}.pdb.gz')
            uncompressed_file = path.join(cache_dir, f'{pdb_id}.pdb')
            # check if uncompressed file already exists
            if not path.exists(uncompressed_file):
                if not path.exists(compressed_file):
                    msg = 'Cannot download .pdb file'
                    return None, msg
                with gzip.open(compressed_file, 'rb') as f_in:
                    with open(uncompressed_file, 'wb') as f_out:
                        copyfileobj(f_in, f_out)

            # modify tnm script template
            with open(path.join(module_dir, 'template-tnm.in'), 'r') as f:
                script_content = f.read()
            replacements = [('PDBID_PLACEHOLDER',   uncompressed_file),
                            ('CHAINID_PLACEHOLDER', str(chain_id_auth)),
                            ('CUTOFF',              str(cutoff))]
            for old, new in replacements:
                script_content = script_content.replace(old, new)

            # save modified script
            with open(script_file, 'w+') as f:
                f.write(script_content)

            # execute the script
            cwd = getcwd()
            f_log = open(tnm_log_file, 'w')
            chdir(cache_dir)
            # need to have tnm software in PATH
            subprocess.run(['tnm', script_file], stdout=f_log)
            chdir(cwd)
            f_log.close()

            # check AGAIN if all files exist
            file_existence = [path.exists(mapping_file)]
            file_existence.append(path.exists(cont_file))
            for dc in ['directionality', 'coordination', 'deformation']:
                file = path.join(cache_dir, f'{prefix}.{dc}_coupling.dat')
                file_existence.append(path.exists(file))
            if not all(file_existence):
                msg = ('TNM cannot execute properly '
                       '(files missing after execution)')
                utils.vprint(verbose, msg)
                return None, msg

        # get residue mapping and number of residues
        with open(mapping_file, 'r') as f:
            lines = f.readlines()
        mapping = [line[:-1].split(' ') for line in lines]

        # indices for dynamical coupling
        dc_idx = [m[0] for m in mapping]

        # list of residue types, indices for contact map, and chain id
        # of each residue
        resnames_1, auth_seq_ids, _ = map(
            list,
            zip(*[m[1].split('_') for m in mapping])
        )

        # keep a list of all resnames encountered in the dataset
        self.all_resnames += resnames_1

        # number of residues
        n_residues = len(lines)

        dc_dict = {dc_idx[i]: i for i in range(n_residues)}
        cont_dict = {auth_seq_ids[i]: i for i in range(n_residues)}

        ################################################################
        # convert TNM results to adjacency matrices
        ################################################################
        utils.vprint(verbose, '  Building graphs Edges...',
                     end='', flush=True)

        # process contact map
        raw_data = np.loadtxt(cont_file, skiprows=1, usecols=(0,1,2),
                              dtype=np.str_)
        cont = np.zeros((n_residues, n_residues), dtype=np.int_)
        for entry in raw_data:
            try:
                i = cont_dict[entry[0]]
                j = cont_dict[entry[1]]
            except KeyError as err:
                msg = f'Random node index reported by TNM software ({err})'
                utils.vprint(verbose, msg)
                return None, msg
            cont[i,j] = 1
            cont[j,i] = 1
        comb = cont.copy() # union of all edges from all dimenisons

        # process dynamical coupling edges
        dc_matrices = {}
        # starting with co-dir (use custom threshold)
        dc = 'directionality'
        file = path.join(cache_dir, f'{prefix}.{dc}_coupling.dat')
        raw_data = np.loadtxt(file, skiprows=3, dtype=np.str_)
        adj_mat = np.zeros((n_residues, n_residues), dtype=np.int_)
        for entry in raw_data:
            if float(entry[2]) > df_codir_thres:
                i = dc_dict[entry[0]]
                j = dc_dict[entry[1]]
                adj_mat[i,j] = 1
                adj_mat[j,i] = 1
        dc_matrices[dc] = adj_mat
        comb += adj_mat
        # # coord and deform (use TNM default threshold)
        # for dc in ['coordination', 'deformation']:
        #     file = path.join(cache_dir, f'{prefix}.{dc}_coupling.dat')
        #     raw_data = np.loadtxt(file, skiprows=3, dtype=np.str_)
        #     adj_mat = np.zeros((n_residues, n_residues), dtype=np.int_)
        #     for entry in raw_data:
        #         i = dc_dict[entry[0]]
        #         j = dc_dict[entry[1]]
        #         adj_mat[i,j] = 1
        #         adj_mat[j,i] = 1
        #     dc_matrices[dc] = adj_mat
        #     comb += adj_mat

        ################################################################
        # convert everything into graphs
        ################################################################
        graph = nx.from_numpy_array(comb)
        graph.graph['pdbID'] = pdb_id.upper()
        if chain_id_rcsb is not None:
            graph.graph['chainID'] = chain_id_rcsb

        utils.vprint(verbose, 'Node Attributes...', end='', flush=True)
        attrs = {i: {'resname': r} for i, r in enumerate(resnames_1)}
        nx.set_node_attributes(graph, attrs)

        # modify edge attributes
        utils.vprint(verbose, 'Edge Attributes...', end='', flush=True)
        for nodeI, nodeJ in graph.edges:
            graph.edges[(nodeI, nodeJ)]['edge_type'] = [0,0]#,0,0]
            if cont[nodeI][nodeJ] == 1: # contact edge
                graph.edges[(nodeI, nodeJ)]['edge_type'][0] = 1
            if dc_matrices['directionality'][nodeI][nodeJ] == 1:
                graph.edges[(nodeI, nodeJ)]['edge_type'][1] = 1
            # if dc_matrices['coordination'][nodeI][nodeJ] == 1:
            #     graph.edges[(nodeI, nodeJ)]['edge_type'][2] = 1
            # if dc_matrices['deformation'][nodeI][nodeJ] == 1:
            #     graph.edges[(nodeI, nodeJ)]['edge_type'][3] = 1
            # remove original weight
            del graph.edges[(nodeI, nodeJ)]['weight']

        # set node indices to match with .pdb file
        mapping = dict(zip(graph, auth_seq_ids)) #atoms.getResnums().tolist()))
        graph = nx.relabel.relabel_nodes(graph, mapping)

        graph_dict = nx.readwrite.json_graph.node_link_data(graph)

        with open(graph_file, 'w+') as f_out:
            json.dump(graph_dict, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        msg = 'Rebuilt' if rebuild_msg else 'Computed'
        utils.vprint(verbose, msg)
        return graph_dict, msg

    def _get_anm_graph(self, pdb_id, chain_id_rcsb, nma_dir,
                       cutoff, n_modes, # gamma, corr_thres,
                       nCPUs, atoms=None, rebuild_existing=False,
                       verbose=None):
        '''
        Returns the contact edges and the coorelation edges from the
        results of NMA.

        Checks if the graph for the specified ID is already saved on
        disk, and returns the data if found. If not found, structure for
        the PDB entry or chain will be downloaded using _get_struct(),
        unless 'atoms' is specified. The arguments will be directed to
        comp_freqs() for modal analysis with ProDy.
        '''

        instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose,
                     f'Retrieving graphs (ANM) for \'{instance_id}\'...',
                     end='', flush=True)

        # define save directory for graphs
        graph_file = path.join(nma_dir, f'{instance_id}.json')

        rebuild_msg = False
        if rebuild_existing and path.exists(graph_file):
            remove_file(graph_file)
            # change output message if entry is rebuilt
            rebuild_msg = True

        # returns data found on disk if rebuild is not required
        if path.exists(graph_file):
            msg = 'Data found on disk'
            with open(graph_file, 'r') as f_in:
                graph_dict = json.load(f_in)
            self.all_resnames += [n['resname'] for n in graph_dict['nodes']]
            utils.vprint(verbose, msg)
            return graph_dict, msg

        # if file was not found
        if not atoms: # use coords if given to save computation
            atoms, msg = self._get_struct(pdb_id, chain_id_rcsb, verbose=False)
            if atoms is None:
                utils.vprint(verbose, msg)
                return None, msg

        utils.vprint(verbose, '  Computing modes...', end='', flush=True)
        anm, freqs, msg = self.comp_freqs(pdb_id, chain_id_rcsb,
                                          atoms=atoms, cutoff=cutoff,
                                          n_modes=n_modes, nCPUs=nCPUs,
                                          verbose=False)
        if not anm:
            msg = 'Unable to compute modes.'
            utils.vprint(verbose, msg)
            return None, msg

        ################################################################
        # compute adjacency matrices based on ANM results from ProDy
        ################################################################
        # compute contact map
        utils.vprint(verbose, 'Kirchhoff...', end='', flush=True)
        cont = - anm.getKirchhoff().astype(np.int_) # not contact map
        np.fill_diagonal(cont, 1) # contact map is completed here

        # compute co-directionality coupling
        utils.vprint(verbose, 'Cross Correlation...', flush=True)
        codir = prody.calcCrossCorr(anm)
        mask = np.abs(codir) > df_codir_thres
        codir = np.where(mask, 1, 0) # correlation map is completed here

        # compute adjacency matrix
        comb = cont + codir

        ################################################################
        # convert adjacency matrices to graphs
        ################################################################
        # create Networkx graph object
        utils.vprint(verbose, '    Building Graphs...',
                     end='', flush=True)
        graph = nx.from_numpy_array(comb)
        graph.graph['pdbID'] = pdb_id.upper()
        if chain_id_rcsb is not None:
            graph.graph['chainID'] = chain_id_rcsb

        # define node attributes
        utils.vprint(verbose, 'Node Attributes...', end='', flush=True)
        resnames = atoms.getResnames()
        attrs = {i: {'resname': r} for i, r in enumerate(resnames)}
        nx.set_node_attributes(graph, attrs)
        # keep a list of all resnames encountered in the dataset
        self.all_resnames += list(resnames)

        # modify edge attributes
        utils.vprint(verbose, 'Edge Attributes...', end='', flush=True)
        for nodeI, nodeJ in graph.edges:
            graph.edges[(nodeI, nodeJ)]['edge_type'] = [0,0]
            if cont[nodeI][nodeJ] == 1: # contact edge
                graph.edges[(nodeI, nodeJ)]['edge_type'][0] = 1
            if codir[nodeI][nodeJ] == 1:
                graph.edges[(nodeI, nodeJ)]['edge_type'][1] = 1
            # if coord[nodeI][nodeJ] == 1:
            #     graph.edges[(nodeI, nodeJ)]['edge_type'][2] = 1
            # if deform[nodeI][nodeJ] == 1:
            #     graph.edges[(nodeI, nodeJ)]['edge_type'][3] = 1
            # remove original weight
            del graph.edges[(nodeI, nodeJ)]['weight']

        # set node indices to match with .pdb file
        mapping = dict(zip(graph, atoms.getResnums().tolist()))
        graph = nx.relabel.relabel_nodes(graph, mapping)

        graph_dict = nx.readwrite.json_graph.node_link_data(graph)

        with open(graph_file, 'w+') as f_out:
            json.dump(graph_dict, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        msg = 'Rebuilt' if rebuild_msg else 'Computed'
        utils.vprint(verbose, msg)
        return graph_dict, msg

    def _update_MFGO_indices(self, processed_ids, save_dir, verbose=True):

        '''
        Updates the list of ID-MFGO saved in preprocessing/stats/target,
        and saves a new copy to preprocessing/stats. The new copy can be
        used for GNN training.
        '''

        utils.vprint(verbose, 'Updating MFGO indices...', end='')
        with open(path.join(self.template_dir, df_mfgo_filename),
                  'r') as f_in:
            id_mfgo = json.load(f_in)

        # fish out all MF-GO entries in processed_ids
        new_id_mfgo = {}
        try:
            if self.entry_type == 'monomer':
                for ID in processed_ids:
                    instance_id = format_pdb_chain_id(ID[:4], ID[5:])
                    new_id_mfgo[instance_id] = id_mfgo[ID]
        except KeyError as err:
            print(f'{err} was not found in '
                  f'{path.join(self.template_dir, df_mfgo_filename)}')
            utils.vprint(verbose, 'MFGO indices will not be updated')
            return None

        mfgo_list = [e for v in new_id_mfgo.values()
                       for e in v]
        new_mfgo_unique, new_mfgo_cnt = np.unique(mfgo_list,
                                                  return_counts=True)

        # discard annotations with too few entries
        if self.go_thres:
            mask = new_mfgo_cnt >= self.go_thres
            if not np.any(mask):
                raise RuntimeError(f'No MF-GOs with over '
                                   f'{self.go_thres} entries')
            new_mfgo_unique = new_mfgo_unique[mask]
            new_mfgo_cnt = new_mfgo_cnt[mask]

        # squeeze the numbering towards 0 (start from 0 continuously)
        for ID in new_id_mfgo:
            labels = []
            for code in new_id_mfgo[ID]:
                loc = np.argwhere(new_mfgo_unique==code)
                if loc.size == 0:
                    continue
                else:
                    labels.append(int(loc[0,0]))
            new_id_mfgo[ID] = labels

        # update mfgo-count file
        cnt_file = path.join(self.template_dir, df_mfgo_cnt_filename)
        if path.exists(cnt_file):
            mfgo_unique = np.loadtxt(cnt_file, dtype=str)[:,0]

            new_cnt = np.column_stack((mfgo_unique[new_mfgo_unique],
                                       new_mfgo_cnt))
        else:
            new_cnt = np.column_stack((np.arange(new_mfgo_unique.size),
                                       new_mfgo_cnt))

        # save count to drive
        np.savetxt(path.join(save_dir, df_mfgo_cnt_filename),
                   new_cnt, fmt='%s %s')

        # check if any labels are present in every data entry
        warning_file = path.join(save_dir, 'warning-mfgo_update.txt')
        if any(new_mfgo_cnt==len(new_id_mfgo)):
            msg = (f'Warning: Labels '
                   f'{new_mfgo_unique[new_mfgo_cnt==len(new_id_mfgo)]} '
                   f'exists in all data entries\n')
            with open(warning_file, 'w+') as f_out:
                f_out.write(msg)
                f_out.flush()
        elif path.exists(warning_file):
            remove_file(warning_file)

        # save id-mfgo
        with open(path.join(save_dir, df_mfgo_filename),
                  'w+') as f_out:
            json.dump(new_id_mfgo, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        utils.vprint(verbose, '')
        ids_without_mfgos = [ID for ID in new_id_mfgo if not new_id_mfgo[ID]]
        np.savetxt(path.join(save_dir, df_noMFGO_filename),
                   ids_without_mfgos, fmt='%s')

        utils.vprint(verbose, 'Done')
        return id_mfgo

    def preprocess(self,
                   # persistence images
                   simplex=df_simplex,
                   # normal mode analysis
                   enm_type='tnm', cutoff=df_cutoff, n_modes=df_n_modes,
                   nCPUs=cpu_count(),
                   retry_download=False, rebuild_pi=False, rebuild_graph=False,
                   update_mfgo=True, verbose=None):

        '''
        Generates all the data needed for training. Call this AFTER
        labels are generated.
        '''

        mfgo_file = path.join(self.template_dir, df_mfgo_filename)
        with open(mfgo_file, 'r') as f_in:
            id_mfgo = json.load(f_in)
        id_list = [ID for ID in id_mfgo]
        print(f'Processing {mfgo_file}')

        verbose = self.verbose if verbose is None else verbose

        self.all_resnames = []

        # holder for PBDs/chains that are successfully preprocessed
        processed_ids = []
        failed_ids = []

        # make directories to save annotation and data
        if enm_type == 'anm':
            nma_setup = anm_setup_folder.format(cutoff, n_modes)
        elif enm_type == 'tnm':
            nma_setup = tnm_setup_folder.format(cutoff)
        else:
            raise NotImplementedError('Choose between anm and tnm for '
                                      'enm type')
        pi_setup = pi_setup_folder.format(simplex)
        go_thres_setup = go_thres_folder.format(self.go_thres)
        # annotation directories
        save_dir = path.join(self.annotations_dir,
                             f'{nma_setup}-{pi_setup}',
                             go_thres_setup)
        makedirs(save_dir, exist_ok=True)
        # data directories
        pi_dir = path.join(df_pi_root, pi_setup)
        nma_dir = path.join(df_graph_root, nma_setup)
        makedirs(pi_dir, exist_ok=True)
        makedirs(nma_dir, exist_ok=True)

        # dataset-specific log
        dataset_log = path.join(save_dir, self.local_process_logname)
        open(dataset_log, 'w+').close() # clear file content

        # backup and clear logfile
        if retry_download and path.exists(self.download_parse_log):
            utils.backup_file(self.download_parse_log)
            # the list of IDs to skip will be empty
            utils.rm_log_entries(self.download_parse_log, id_list)

        # get list of PDBs/chains that should be skipped
        log_content, logged_ids = utils.read_logs(self.download_parse_log)
        utils.vprint(verbose,
                     f' -> {len(logged_ids)} PDB entries found in log, '
                     f'{len(set(logged_ids) & set(id_list))} to be skipped')

        for ID in tqdm(id_list, unit=' entries',
                       desc='Processing data',
                       ascii=True, dynamic_ncols=True):

            pdb_id = ID[:4]
            chain_id_rcsb = ID[5:] if len(ID)>5 else None
            instance_id = format_pdb_chain_id(pdb_id, chain_id_rcsb)

            # ############################################################
            # # skip everything if all data for ID is found on disk
            # ############################################################
            # if not rebuild_pi and not rebuild_graph:
            #     pi_file = path.join(pi_dir, f'{instance_id}.npy')
            #     graph_file = path.join(nma_dir, f'{instance_id}.json')
            #     if path.exists(graph_file) and path.exists(pi_file):
            #         processed_ids.append(instance_id)
            #         tqdm.write(f'  All data for \'{instance_id}\' '
            #                    f'found on disk.')
            #         continue

            ############################################################
            # if the PDB/chain failed to download in a previous run
            ############################################################
            if instance_id in logged_ids:
                # copy entry to dataset-specific log
                idx = logged_ids.index(instance_id)
                utils.append_to_file(log_content[idx], dataset_log)
                failed_ids.append(instance_id)
                tqdm.write(f'  Skipping processing of \'{instance_id}\'')
                continue

            # if the PDB entry was not skipped
            tqdm.write(f'  Processing \'{instance_id}\'...')

            ############################################################
            # try to download/parse structure
            ############################################################
            tqdm.write('    Download/Parsing Structure...')
            atoms, msg = self._get_struct(pdb_id, chain_id_rcsb, verbose=False)
            tqdm.write(f'      {msg}')
            if atoms is None:
                # write entry to dataset-specific log
                utils.append_to_file(f'{instance_id} -> ProDy: {msg}',
                                     dataset_log)
                # write new entry to log for all datasets
                utils.append_to_file(f'{instance_id} -> ProDy: {msg}',
                                     self.download_parse_log)
                failed_ids.append(instance_id)
                continue
            coords = atoms.getCoords().tolist()

            ############################################################
            # try to generate persistence image
            ############################################################
            tqdm.write('    Persistence Img...')
            pers_img, msg = self._get_PI(pdb_id, chain_id_rcsb,
                                         pi_dir,
                                         coords=coords,
                                         simplex=simplex,
                                         rebuild_existing=rebuild_pi,
                                         verbose=False)
            tqdm.write(f'      {msg}')
            if pers_img is None:
                # write entry to dataset-specific log
                utils.append_to_file(f'{instance_id} -> PI: {msg}',
                                     dataset_log)
                failed_ids.append(instance_id)
                continue

            ############################################################
            # try to generate graphs (normal mode analysis)
            ############################################################
            tqdm.write(f'    Graph ({enm_type.upper()})...')

            if enm_type.lower() == 'anm':
                graph_dict, msg = self._get_anm_graph(
                    pdb_id, chain_id_rcsb, nma_dir,
                    atoms=atoms, cutoff=cutoff, n_modes=n_modes,
                    nCPUs=nCPUs, rebuild_existing=rebuild_graph, verbose=False
                )
            elif enm_type.lower() == 'tnm':
                graph_dict, msg = self._get_tnm_graph(
                    pdb_id, chain_id_rcsb, nma_dir,
                    cutoff=cutoff, nCPUs=nCPUs,
                    rebuild_existing=rebuild_graph, verbose=False
                )
            tqdm.write(f'      {msg}')

            if graph_dict is None:
                # write entry to dataset-specific log
                utils.append_to_file(f'{instance_id} -> NMA ({enm_type}): '
                                     f'{msg}', dataset_log)
                failed_ids.append(instance_id)
                continue

            ############################################################
            # all computations successful
            ############################################################
            processed_ids.append(instance_id)

        # save a list of all residues encountered in dataset
        uni, cnt = np.unique(self.all_resnames, return_counts=True)
        np.savetxt(path.join(save_dir, 'all_resnames.txt'),
                   np.column_stack((uni, cnt)), fmt='%s %s')

        print('Saving...') #, end='')
        if self.entry_type == 'monomer':
            # save successful ids
            np.savetxt(path.join(save_dir, df_chain_filename),
                       processed_ids, fmt='%s')
            successful_pdb = [e[:4] for e in processed_ids]
            np.savetxt(path.join(save_dir, df_pdb_filename),
                       successful_pdb, fmt='%s')

            # save unsuccessful ids
            np.savetxt(path.join(save_dir,
                                 df_failed_chain_filename),
                       failed_ids, fmt='%s')
            unsuccessful_pdb = [e[:4] for e in failed_ids]
            np.savetxt(path.join(save_dir, df_failed_pdb_filename),
                       unsuccessful_pdb, fmt='%s')

        print('  Done')

        # output summary
        print(f' -> {len(processed_ids)} out of {len(id_list)} '
              f'entries successfully processed '
              f'({len(failed_ids)} entries failed)')

        # update label files
        if update_mfgo:
            print('Updating MFGO label files...')
            id_mfgo = self._update_MFGO_indices(processed_ids, save_dir,
                                                verbose=False)
            if id_mfgo:
                print('  Done')
            else:
                print('  Update aborted')

        print('>>> Preprocessing Complete')

    def cleanup(self):

        while True:
            inp = input(f'All preprocessed data not in {self.set_name} '
                        f'will be removed. Proceed? (y/n)')
            if inp.lower() in ['y', 'n']:
                break
