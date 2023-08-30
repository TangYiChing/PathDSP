#!/homes/ac.rgnanaolivu/miniconda3/envs/rohan_python/bin/python

import sys
import os
import numpy as np
import polars as pl
#import torch
#import torch.utils.data as du
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.nn.functional as F
#from code.drugcell_NN import *
import argparse
import numpy as np
import pandas as pd
import candle
#import time
#import logging
#import networkx as nx
#import networkx.algorithms.components.connected as nxacc
#import networkx.algorithms.dag as nxadag
#from pathlib import Path
from functools import reduce
import improve_utils
# import RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from datetime import datetime
# import NetPEA modules
import RWR as rwr
import NetPEA as pea
#import gsea module
import gseapy as gp




file_path = os.path.dirname(os.path.realpath(__file__))
#fdir = Path('__file__').resolve().parent
#source = 'csa_data/raw_data/splits/'
required = None
additional_definitions = None

# This should be set outside as a user environment variable
#os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'

# initialize class
class PathDSP_candle(candle.Benchmark):
    def set_locals(self):
        '''
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        '''
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    preprocessor_bmk = PathDSP_candle(file_path,
        'PathDSP_params.txt',
        'pytorch',
        prog='PathDSP_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters


def mkdir(directory):
    directories = directory.split('/')   

    folder = ''
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)


def preprocess(params, data_dir):
    print(os.environ['CANDLE_DATA_DIR'])
    #requirements go here
    #keys_parsing = ['output_dir', 'hidden', 'result', 'metric', 'data_type']
    if not os.path.exists(data_dir):
        mkdir(data_dir)
    params['data_dir'] = data_dir
    #args = candle.ArgumentStruct(**params)
    for i in ['train_data', 'test_data', 'val_data', 'drug_bits_file', 'dgnet_file', 
              'mutnet_file', 'cnvnet_file', 'exp_file']:
        params[i] = params['data_dir'] + '/' + params[i]
    return(params)

def download_anl_data(params):
    csa_data_folder = os.path.join(os.environ['CANDLE_DATA_DIR'], 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')

    if not os.path.exists(csa_data_folder):
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
        mkdir(splits_dir)
        mkdir(x_data_dir)
        mkdir(y_data_dir)

    for improve_file in ['CCLE_all.txt', 
                         'CCLE_split_' + str(params['split']) + '_test.txt',
                         'CCLE_split_' + str(params['split']) + '_train.txt',
                         'CCLE_split_' + str(params['split']) + '_val.txt',
                         'CTRPv2_all.txt', 
                         'CTRPv2_split_' + str(params['split']) + '_test.txt',
                         'CTRPv2_split_' + str(params['split']) + '_train.txt',
                         'CTRPv2_split_' + str(params['split']) + '_val.txt',
                         'gCSI_all.txt',
                         'GDSCv1_all.txt',
                         'GDSCv2_all.txt'
                         ]:
        url_dir = params['improve_data_url'] + '/splits/' 
        candle.file_utils.get_file(improve_file, url_dir + improve_file,
                                   datadir=splits_dir,
                                   cache_subdir=None)

    for improve_file in ['cancer_mutation_count.tsv', 'drug_SMILES.tsv', 'drug_info.tsv', 'cancer_discretized_copy_number.tsv', 'cancer_gene_expression.tsv']:
        url_dir = params['improve_data_url'] + '/x_data/' 
        candle.file_utils.get_file(fname=improve_file, origin=url_dir + improve_file,
                                   datadir=x_data_dir,
                                   cache_subdir=None)

    url_dir = params['improve_data_url'] + '/y_data/'
    response_file  = 'response.tsv'
    candle.file_utils.get_file(fname=response_file, origin=url_dir + response_file,
                                   datadir=y_data_dir,
                                   cache_subdir=None)
    
    ## get gene-set data and string data
    for db_file in [params['gene_set'], params['ppi_data'], params['drug_target']]:
        candle.file_utils.get_file(db_file, params['data_url'] + '/' +db_file,
                                   datadir=params['data_dir'],
                                   cache_subdir=None)
        

    
    
# set timer
def cal_time(end, start):
    '''return time spent'''
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start),datetimeFormat)
    return spend


def download_author_data(params):
    data_download_filepath = candle.get_file(params['original_data'], params['original_data_url'] + '/' + params['original_data'],
                                             datadir = params['data_dir'],
                                             cache_subdir = None)
    print('download_path: {}'.format(data_download_filepath))


def smile2bits(params):
    start = datetime.now()
    rs_all = improve_utils.load_single_drug_response_data(source=params['data_type'],
                                                         split=params['split'], split_type=["train", "test", "val"],
                                                         y_col_name=params['metric'])
    smile_df = improve_utils.load_smiles_data()
    smile_df.columns = ['drug', 'smile']
    smile_df = smile_df.drop_duplicates(subset=['drug'], keep='first').set_index('drug')
    smile_df = smile_df.loc[smile_df.index.isin(rs_all['improve_chem_id']),]
    bit_int = params['bit_int']
    record_list = []
    # smile2bits drug by drug
    n_drug = 1
    for idx, row in smile_df.iterrows():
        drug = idx
        smile = row['smile']
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        mbit = list( AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=bit_int) )
        #drug_mbit_dict.update({drug:mbit})
        # append to result
        record_list.append( tuple([drug]+mbit) )
        if len(mbit) == bit_int:
            n_drug+=1
    print('total {:} drugs with bits'.format(n_drug))
    # convert dict to dataframe
    colname_list = ['drug'] + ['mBit_'+str(i) for i in range(bit_int)]
    drug_mbit_df = pd.DataFrame.from_records(record_list, columns=colname_list)
    #drug_mbit_df = pd.DataFrame.from_dict(drug_mbit_dict, orient='index', columns=colname_list)
    #drug_mbit_df.index.name = 'drug'
    print('unique drugs={:}'.format(len(drug_mbit_df['drug'].unique())))
    # save to file
    drug_mbit_df.to_csv(params['drug_bits_file'], header=True, index=False, sep='\t')
    print('[Finished in {:}]'.format(cal_time(datetime.now(), start)))

def times_expression(rwr, exp):
    '''
    :param rwrDf: dataframe of cell by gene probability matrix
    :param expDf: dataframe of cell by gene expression matrix
    :return rwr_timesexp_df: dataframe of cell by gene probability matrix,
                             in which genes are multiplied with expression values

    Note: this function assumes cells are all overlapped while gene maybe not
    '''
    cell_list = sorted(list(set(rwr.index) & set(exp.index)))
    gene_list = sorted(list(set(rwr.columns)&set(exp.columns)))

    if len(cell_list) == 0:
        print('ERROR! no overlapping cell lines')
        sys.exit(1)
    if len(gene_list) == 0:
        print('ERROR! no overlapping genes')
        sys.exit(1)

    # multiply with gene expression for overlapping cell, gene
    rwr_timesexp =  rwr.loc[cell_list, gene_list]*exp.loc[cell_list, gene_list]

    # concat with other gene
    out_gene_list = list(set(rwr.columns)-set(gene_list))
    out_df = pd.concat([rwr_timesexp, rwr[out_gene_list]], axis=1)
    return out_df

def run_netpea(params, dtype, multiply_expression):
    # timer
    start_time = datetime.now()
    ppi_path = params['data_dir'] + '/STRING/9606.protein_name.links.v11.0.pkl'
    pathway_path = params['data_dir'] + '/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt'
    log_transform = False
    permutation_int = params['permutation_int']
    seed_int = params['seed_int']
    cpu_int = params['cpu_int']
    csa_data_folder = os.path.join(os.environ['CANDLE_DATA_DIR'], 'csa_data', 'raw_data')
    rs_all = improve_utils.load_single_drug_response_data(source=params['data_type'],
                                                        split=params['split'], split_type=["train", "test", "val"],
                                                        y_col_name=params['metric'])
    if dtype == 'DGnet':
        drug_info = pd.read_csv(csa_data_folder + '/x_data/drug_info.tsv', sep='\t')
        drug_info['NAME'] = drug_info['NAME'].str.upper()
        target_info = pd.read_csv(params['data_dir'] + '/data/DB.Drug.Target.txt', sep = '\t')
        target_info = target_info.rename(columns={'drug': 'NAME'})
        combined_df = pd.merge(drug_info, target_info, how = 'left', on = 'NAME').dropna(subset=['gene'])
        combined_df = combined_df.loc[combined_df['improve_chem_id'].isin(rs_all['improve_chem_id']),]
        restart_path = params['data_dir'] + '/drug_target.txt'
        combined_df.iloc[:,-2:].to_csv(restart_path, sep = '\t', header= True, index=False)
        outpath = params['dgnet_file']
    elif dtype == 'MUTnet':
        mutation_data = improve_utils.load_mutation_count_data(gene_system_identifier='Gene_Symbol')
        mutation_data = mutation_data.reset_index()
        mutation_data = pd.melt(mutation_data, id_vars='improve_sample_id').loc[lambda x: x['value'] > 0]
        mutation_data = mutation_data.loc[mutation_data['improve_sample_id'].isin(rs_all['improve_sample_id']),]
        restart_path = params['data_dir'] + '/mutation_data.txt'
        mutation_data.iloc[:,0:2].to_csv(restart_path, sep = '\t', header= True, index=False)
        outpath = params['mutnet_file']
    else:
        cnv_data = improve_utils.load_discretized_copy_number_data(gene_system_identifier='Gene_Symbol')
        cnv_data = cnv_data.reset_index()
        cnv_data = pd.melt(cnv_data, id_vars='improve_sample_id').loc[lambda x: x['value'] != 0]
        cnv_data = cnv_data.loc[cnv_data['improve_sample_id'].isin(rs_all['improve_sample_id']),]
        restart_path = params['data_dir'] + '/cnv_data.txt'
        cnv_data.iloc[:,0:2].to_csv(restart_path, sep = '\t', header= True, index=False)
        outpath = params['cnvnet_file']
    # perform Random Walk
    print(datetime.now(), 'performing random walk with restart')
    rwr_df = rwr.RWR(ppi_path, restart_path, restartProbFloat=0.5, convergenceFloat=0.00001, normalize='l1', weighted=True).get_prob()
    # multiply with gene expression
    if multiply_expression:
        print(datetime.now(), 'multiplying gene expression with random walk probability for genes were expressed')
        exp_df = improve_utils.load_gene_expression_data(gene_system_identifier='Gene_Symbol')
        rwr_df = times_expression(rwr_df, exp_df)
    #rwr_df.to_csv(out_path+'.RWR.txt', header=True, index=True, sep='\t')
    # perform Pathwa Enrichment Analysis
    print(datetime.now(), 'performing network-based pathway enrichment')
    cell_pathway_df = pea.NetPEA(rwr_df, pathway_path, log_transform=log_transform, permutation=permutation_int, seed=seed_int, n_cpu=cpu_int, out_path=outpath)
    print( '[Finished in {:}]'.format(cal_time(datetime.now(), start_time)) )
    
def prep_input(params):
    # Read data files
    drug_mbit_df = pd.read_csv(params['drug_bits_file'], sep = '\t', index_col=0)
    drug_mbit_df = drug_mbit_df.reset_index().rename(columns={'drug': 'drug_id'})
    DGnet = pd.read_csv(params['dgnet_file'], sep='\t', index_col=0)
    DGnet = DGnet.add_suffix('_dgnet').reset_index().rename(columns={'index': 'drug_id'})
    CNVnet = pd.read_csv(params['cnvnet_file'], sep= '\t',index_col=0)
    CNVnet = CNVnet.add_suffix('_cnvnet').reset_index().rename(columns={'index': 'sample_id'})
    MUTnet = pd.read_csv(params['mutnet_file'], sep='\t',index_col=0)
    MUTnet = MUTnet.add_suffix('_mutnet').reset_index().rename(columns={'index': 'sample_id'})
    EXP = pd.read_csv(params['exp_file'], sep = '\t', index_col=0)
    EXP = EXP.add_suffix('_exp').reset_index().rename(columns={'index': 'sample_id'})
    response_df = improve_utils.load_single_drug_response_data(source=params['data_type'], split=params['split'],
                                                            split_type=['train', 'test', 'val'],
                                                            y_col_name= params['metric'])
    response_df = response_df.rename(columns={'improve_chem_id': 'drug_id', 'improve_sample_id': 'sample_id'})
    # Extract relevant IDs

    common_drug_ids = reduce(np.intersect1d, (drug_mbit_df['drug_id'], DGnet['drug_id'], response_df['drug_id']))
    common_sample_ids = reduce(np.intersect1d, (CNVnet['sample_id'], MUTnet['sample_id'], EXP['sample_id'] , response_df['sample_id']))
    response_df = response_df.loc[(response_df['drug_id'].isin(common_drug_ids)) & 
                            (response_df['sample_id'].isin(common_sample_ids)), :]
    drug_mbit_df = drug_mbit_df.loc[drug_mbit_df['drug_id'].isin(common_drug_ids), :].set_index('drug_id').sort_index()
    DGnet = DGnet.loc[DGnet['drug_id'].isin(common_drug_ids), :].set_index('drug_id').sort_index()
    CNVnet = CNVnet.loc[CNVnet['sample_id'].isin(common_sample_ids), :].set_index('sample_id').sort_index()
    MUTnet = MUTnet.loc[MUTnet['sample_id'].isin(common_sample_ids), :].set_index('sample_id').sort_index() 
    EXP = EXP.loc[EXP['sample_id'].isin(common_sample_ids), :].set_index('sample_id').sort_index()
    
    drug_data = drug_mbit_df.join(DGnet)
    sample_data = CNVnet.join([MUTnet, EXP])
    ## export train,val,test set
    for i in ['train', 'test', 'val']:
        response_df = improve_utils.load_single_drug_response_data(source=params['data_type'], split=params['split'],
                                                            split_type=i,
                                                            y_col_name= params['metric'])
        response_df = response_df.rename(columns={'improve_chem_id': 'drug_id', 'improve_sample_id': 'sample_id'})
        response_df = response_df.loc[(response_df['drug_id'].isin(common_drug_ids)) & 
                        (response_df['sample_id'].isin(common_sample_ids)), :]
        comb_data_mtx = pd.DataFrame({'drug_id': response_df['drug_id'].values, 
                            'sample_id': response_df['sample_id'].values})
        comb_data_mtx = comb_data_mtx.set_index(['drug_id', 'sample_id']).join(drug_data, on = 'drug_id').join(sample_data, on = 'sample_id')
        comb_data_mtx['response'] = response_df[params['metric']].values
        comb_data_mtx = comb_data_mtx.dropna()
        pl.from_pandas(comb_data_mtx).write_csv(params[i + '_data'], separator = '\t', has_header = True)


def run_ssgsea(params):
    expMat = improve_utils.load_gene_expression_data(sep='\t')
    rs_all = improve_utils.load_single_drug_response_data(source=params['data_type'],
                                                        split=params['split'], split_type=["train", "test", "val"],
                                                        y_col_name=params['metric'])
    expMat = expMat.loc[expMat.index.isin(rs_all['improve_sample_id']),]
    gct = expMat.T # gene (rows) cell lines (columns)
    pathway_path = params['data_dir'] + '/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt'
    gmt = pathway_path
    tmp_str = params['data_dir']

    if not os.path.isdir(tmp_str):
        os.mkdir(tmp_str) 

    # run enrichment
    ssgsea = gp.ssgsea(data=gct,  #gct: a matrix of gene by sample
                           gene_sets=gmt, #gmt format
                           outdir=tmp_str,
                           scale=True,
                           permutation_num=0, #1000
                           no_plot=True,
                           processes=params['cpu_int'],
                           #min_size=0,
                           format='png')

    result_mat = ssgsea.res2d.T # get the normalized enrichment score (i.e., NES)
    result_mat.to_csv(tmp_str+'ssGSEA.txt', header=True, index=True, sep="\t")

    f = open(tmp_str+'ssGSEA.txt', 'r')
    lines = f.readlines()
    total_dict = {}
    for cell in set(lines[1].split()):
        total_dict[cell] = {}
    cell_lines = lines[1].split()
    vals = lines[4].split()
    for i, pathway in enumerate((lines[2].split())):
        if i > 0:
            total_dict[cell_lines[i]][pathway] = float(vals[i])
    df = pd.DataFrame(total_dict)
    df.T.to_csv(params['exp_file'], header=True, index=True, sep="\t")


def candle_main(anl):
    params = initialize_parameters()
    data_dir = os.environ['CANDLE_DATA_DIR'] + '/' + '/Data/'
    params =  preprocess(params, data_dir)
    if params['improve_analysis'] == 'yes' or anl:
        download_anl_data(params)
        print('convert drug to bits.')
        smile2bits(params)
        print('compute DGnet.')
        run_netpea(params, dtype = 'DGnet', multiply_expression=False)
        print('compute MUTnet.')
        run_netpea(params, dtype = 'MUTnet', multiply_expression=True)
        print('compute CNVnet.')
        run_netpea(params, dtype = 'CNVnet', multiply_expression=True)
        print('compute EXP.')
        run_ssgsea(params)
        print('prepare final input file.')
        prep_input(params)
    else:
        download_author_data(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-a', dest='anl',  default=False)
    args = parser.parse_args()
    start = datetime.now()
    candle_main(args.anl)
    print('[Finished in {:}]'.format(cal_time(datetime.now(), start)))
