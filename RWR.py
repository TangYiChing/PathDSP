"""
Return cell by gene probability dataframe

"""


import argparse
import numpy as np
import pandas as pd
import scipy.sparse as scisp
import sklearn.preprocessing as skprc
from datetime import datetime


class RWR:
    """
    Return probability matrix where columns are PPI genes

    :param ppiPathStr: string representing path to ppi file (with three columns)
    :param restartPathStr: string representing path to restart file (i.e., input gene sets)
    :param restartProbFloat: float representing restart probability (default: 0.5)
    :param convergenceFloat: folat representing convergence criterion (default: 1e-5)
    :param normalize: string representing normalization method (choices=['l1', 'l2'])
    :param weighted: boolean indicating weither to use weighted graph or not (if False, will set weight of all edges to 1)
    :param outPathStr: string representing output path  
    """
    def __init__(self, ppiPathStr, restartPathStr, restartProbFloat=0.5, convergenceFloat=0.00001, normalize='l1', weighted=True, outPathStr='./'):
        # initiating
        self.ppiPathStr = ppiPathStr
        self.restartPathStr = restartPathStr
        self.restartProbFloat = float(restartProbFloat)
        self.convergenceFloat = float(convergenceFloat)
        self.normalize = normalize
        self.weighted = weighted
        self.outPathStr = outPathStr

       

    def get_prob(self):
        # load PPI graph
        print('loading protein-protein interaction network.....')
        self.adj_mat, self.name_idx_dict = self.load_graph(self.ppiPathStr, normalize=True, weighted=True)
        # mapping dictionary of node index number: node name string
        self.idx_name_dict = { idx:name for name, idx in self.name_idx_dict.items() }
 
        # load restart list (i.e., input gene sets)
        print('collecting restart list')
        df = pd.read_csv(self.restartPathStr, header=0, sep="\t")
        df.columns = ['group', 'gene']
        # collect gene sets by group
        grps = df.groupby('group')
        grps_dict = {}
        for grp in df['group'].unique():
            seed_list = grps.get_group(grp)['gene'].values.tolist() #input gene set
            # check if input gene set in ppi and convert name to index number
            seed_idx_list = [self.name_idx_dict[i] for i in seed_list if i in self.name_idx_dict.keys()]
            # update to dictionary
            grps_dict.update({ grp: {'gList':seed_list, 'ppiList':seed_idx_list} })
        
        # perform random walk 
        print('performing random walk.....')
        n_grps = len(grps_dict)
        grp_list = list(grps_dict.keys())
        grp_prob_dict = {}
        n_grp_has_no_ppiList = 0 # number of group has restart list not found on PPI network
        for i in range(n_grps):
            grp = grp_list[i]
            if len(grps_dict[grp]['ppiList']) > 0: # has restart list on PPI network
                prob_list = self.run_single_rwr(self.adj_mat, grps_dict[grp]['ppiList'], restartProbFloat=self.restartProbFloat, convergenceFloat=self.convergenceFloat)
            
            else:
                n_grp_has_no_ppiList += 1
                prob_list = [0.0] * len(self.name_idx_dict)

            # update to result
            grp_prob_dict.update( {grp:prob_list} )

        # reformat result: dict2fataframe
        print('finalizing result of probability matrix.....')
        result_df = pd.DataFrame(grp_prob_dict)
        result_df = result_df.T
        result_df.columns = list(self.name_idx_dict.keys())
        return result_df # probability matrix grp by ppi genes


    def load_graph(self, ppiPathStr, normalize=True, weighted=True):
        """
        Return a graph in adjacency matrix format and its name string and correspoing index number mapping dictionary

        :param ppiPathStr: string representing file name of a graph in edge list format
        :param name2index: boolean indicating whether to convert name string to index number or not
        :param normalize: boolean indicating whether to perform column-wised normalization
        """
        # load data
        df = pd.read_pickle(ppiPathStr) 
        df.columns = ['source', 'target', 'weight']
    
        # convert name to index
        all_nodes = sorted(list(set( df['source'] ) | set( df['target'] ))) # retrieve name strings of all nodes
    
        # create name:index mapping dictionary
        gnm_gid_dict = { all_nodes[i]:i for i in range(len(all_nodes)) }
    
        # replace name string with index number
        df['source'].update(df['source'].map(gnm_gid_dict))
        df['target'].update(df['target'].map(gnm_gid_dict))
    
        # use weighted graph or unweighted graph
        if weighted == False:
            df['weight'] = 1 # unweighted graph
    
        # create adjancency matrix
        network_matrix = scisp.csr_matrix((df['weight'].values, (df['source'].values, df['target'].values)),
                                           shape=(len(all_nodes), len(all_nodes)), dtype=float) # Create sparse matrix
        network_matrix = (network_matrix + network_matrix.T) # Make the ajdacency matrix symmetric
        network_matrix.setdiag(0) # Set diagnoals to zero
    
        # normalization: Normalize the rows of network_matrix because we are multiplying vector by matrix (from left)
        if normalize == True:
            network_matrix = skprc.normalize(network_matrix, norm='l1', axis=1)
    
        # return 
        return network_matrix, gnm_gid_dict

    def run_single_rwr(self, ppiAdjMat, restartList, restartProbFloat=0.5, convergenceFloat=0.00001):
        """
        Return

        :param ppiAdjMat: adjacency matrix of protein-protein interaction network
        :param restartList: list of restart nodes (i.e., gene list)
        :param restartProbFloat: float representing restart probability (default: 0.5)
        :param convergenceFloat: folat representing convergence criterion (default: 1e-5)
        """
        # settings
        convergence_criterion_float = float(convergenceFloat)   # stops when vector L1 norm drops below 10^(-5)
        restartProbFloat = float(restartProbFloat)
        residual_float = 1.0 # difference between p^(t + 1) and p^(t)
        max_iter = 1000

        # initialze probability vector for restart nodes
        prob_seed_list = [0] * ppiAdjMat.shape[0]
        for idx in restartList:
            prob_seed_list[idx] = 1.0 #1/float(len(restartList))
        prob_seed_arr = np.array(prob_seed_list)
        steady_prob_old = prob_seed_arr

        # RWR
        iter_int = 0
        #print('updating probability array.....')
        while (residual_float > convergence_criterion_float):
            # update vector
            steady_prob_new = scisp.csr_matrix.dot(steady_prob_old, ppiAdjMat)
            steady_prob_new *= (1 - restartProbFloat)
            steady_prob_new += (prob_seed_arr * restartProbFloat)

            # Calculate the residual -- the sum of the absolute
            # differences of the new node probability vector minus the old
            # diff_norm = np.linalg.norm(np.subtract(p_t_1, p_t), 1)
            residual_float = abs(steady_prob_new - steady_prob_old).sum()
            steady_prob_old = steady_prob_new.copy()
        return steady_prob_old

