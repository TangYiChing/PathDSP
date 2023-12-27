"""
Implementation of NetPEA: pathway enrichment with networks (Liu, 2017)

Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664096/
zscore >1.65, equivalent to p-value=0.05
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import scipy.stats as scistat
from datetime import datetime

class NetPEA:
    """
    :param rwrDf: dataframe with cell by PPI genes
    :param pathwayGMT: pathway database in gmt format
    :param permutation:
    :param seed:
    :param threshold:
    """
    def __init__(self, rwrPath, pathwayGMT, log_transform=False, permutation=1000, seed=42, n_cpu=5, out_path='./'):
        # load data
        self.rwr_path = rwrPath #pd.read_csv(rwrDf, header=0, index_col=0, sep="\t")
        self.pathway_gmt = pathwayGMT
        self.permutation = int(permutation)
        self.seed = int(seed)
        self.out_path = out_path

        # settings
        np.random.seed(self.seed)
        self.n_cpu = int(n_cpu)
        if len(self.rwr_path) < self.n_cpu:
            self.n_cpu = len(self.rwr_path)

        # prepare pathway genes to save time
        print('{:}: collect pathway genes'.format(datetime.now()))
        pathway_geneList_dict = self._get_pathway_genes(pathwayGMT) # {pathway: geneList}
        # obtain shared genes for calculating score of pathway genes
        self.rwrDf = self.rwr_path#pd.read_csv(rwrPath, header=0, index_col=0, sep="\t")
        if log_transform == True:
            print('log transform input data')
            self.rwrDf = np.log(self.rwrDf)    
        pathway_shareGeneList_dict = self._find_overlaps(self.rwrDf, pathway_geneList_dict) # {pathway: shareGeneList}
        # generate random gene list for calculating score of random pathway genes
        pathway_randomGeneListList_dict = {}
        bg_gene_list = self.rwrDf.columns.tolist() # ppi genes
        for pathway, shareGeneList in pathway_shareGeneList_dict.items():
            pathway_randomGeneListList_dict.update({pathway:[]})
            for  p in range(self.permutation):
                gene_list = np.random.choice(bg_gene_list, len(shareGeneList)).tolist()
                pathway_randomGeneListList_dict[pathway].append(gene_list)
        self.pathwayDictList = [pathway_geneList_dict, pathway_shareGeneList_dict, pathway_randomGeneListList_dict]

        # call function
        self.netpea_parallel(self.rwrDf, self.pathwayDictList, self.n_cpu, self.out_path)

    def netpea_parallel(self, rwrDf, pathwayDictList, n_cpu, out_path):
        # split dataframe
        n_partitions = int(n_cpu)
        split_list = np.array_split(rwrDf, n_partitions)
        # parallel computing
        pool = mp.Pool(int(n_cpu))
        df_list =  pool.starmap(self.netpea, [(df, pathwayDictList) for df in split_list])
        pool.close()
        pool.join()
        print('{:}: comple {:} dfs'.format(datetime.now(), len(df_list)))
        print(df_list[0])

        # merge result of all cells and save to file
        print('{:}: merge result of all cells and save to file'.format(datetime.now()))
        all_cell_zscore_df = pd.concat(df_list, axis=0)
        zscore_fname = self.out_path
        all_cell_zscore_df.to_csv(zscore_fname, header=True, index=True, sep="\t")
        #print(all_cell_zscore_df)


    def netpea(self, rwrDf, pathwayDictList):
        """return dataframe with cell by pathway"""
        pathway_geneList_dict, pathway_shareGeneList_dict, pathway_randomGeneListList_dict = pathwayDictList
        # convert to dataframe with headers=[pathway, #pathway genes, overlap genes]
        pathway_df = self._merge_pathway_dict(pathway_geneList_dict, pathway_shareGeneList_dict)
        # collect score of random gene list
        print('{:}: collect score of random gene list'.format(datetime.now()))
        cell_pathway_bgScoreList_dict = {} # dict of dict
        for cell in rwrDf.index:
            cell_pathway_bgScoreList_dict.update({cell:{}})
            # prepare data
            rwr_df = rwrDf.loc[cell] # 1 by ppiG dataframe
            # append aggregate score for each randomgenelist for each pathway
            for pathway, randomGeneListList in pathway_randomGeneListList_dict.items():
                bgScoreList = [rwr_df.loc[randomGeneList].mean() for randomGeneList in randomGeneListList]
                cell_pathway_bgScoreList_dict[cell].update({pathway:bgScoreList})

        # collect score of share gene list
        print('{:}: collect score of share gene list'.format(datetime.now()))
        cell_pathway_ScoreList_dict = {} # dict of dict
        for cell in rwrDf.index:
            cell_pathway_ScoreList_dict.update({cell:{}})
            # prepare data
            rwr_df = rwrDf.loc[cell] # 1 by ppiG dataframe
            # append aggregate score for each randomgenelist for each pathway
            for pathway, shareGeneList in pathway_shareGeneList_dict.items():
                score = rwr_df.loc[shareGeneList].mean()
                cell_pathway_ScoreList_dict[cell].update({pathway:score})
        # ztest to determin significance
        print('{:}: ztest to determin significance'.format(datetime.now()))
        zscore_dfs = []
        cell_pathway_zscore_dict = {} # collect zscore for each pathway
        cell_pathway_ztest_dict = {} # collect zscore and pvalue for each pathway
        for cell in rwrDf.index:
            cell_pathway_zscore_dict.update({cell:{}})
            cell_pathway_ztest_dict.update({cell:{}})
            pathway_score_dict = cell_pathway_ScoreList_dict[cell]
            pathway_bgList_dict = cell_pathway_bgScoreList_dict[cell]
            for pathway in pathway_geneList_dict.keys():
                score = pathway_score_dict[pathway]
                bgList = pathway_bgList_dict[pathway]
                [zscore, pvalue] = self._cal_zscore(score, bgList)
                cell_pathway_ztest_dict[cell].update({pathway: [zscore, pvalue]})
                cell_pathway_zscore_dict[cell].update({pathway:zscore})
            # save per-cell zscore 
            cell_zscore_df = pd.DataFrame(cell_pathway_zscore_dict[cell], index=[cell])
            zscore_dfs.append(cell_zscore_df)
            # save per-cell ztest results 
            cell_bgtest_df = pd.DataFrame(cell_pathway_ztest_dict[cell], index=['zscore', 'pvalue']).T
            cell_bgtest_df.index.name = 'pathway'
            cell_bgtest_df = cell_bgtest_df.join(pathway_df)
            #percell_fname = self.out_path + '.' + cell + '.NetPEA.background_result.txt'
            #cell_bgtest_df.to_csv(percell_fname, header=True, index=True, sep="\t")
        # merge result of all cells and save to file
        #print('{:}: merge result of all cells and save to file'.format(datetime.now()))
        all_cell_zscore_df = pd.concat(zscore_dfs, axis=0)
        #zscore_fname = self.out_path + '.NetPEA.zscore.txt'
        #all_cell_zscore_df.to_csv(zscore_fname, header=True, index=True, sep="\t")
       
        # clear space
        pathwayDictList = []
        return all_cell_zscore_df

    def _merge_pathway_dict(self, pathway_geneList_dict, pathway_shareGeneList_dict):
        """return dataframe with headers = [pathway, #pathway genes, overlap genes]"""
        pathway_lenG_dict = {pathway: len(geneList) for pathway, geneList in pathway_geneList_dict.items()}
        pathway_strG_dict = {pathway: ",".join(geneList) for pathway, geneList in pathway_shareGeneList_dict.items()}
        df1 = pd.DataFrame(pathway_lenG_dict.items(), columns=['pathway', '#pathway genes'])
        df2 = pd.DataFrame(pathway_strG_dict.items(), columns=['pathway', 'overlap genes'])
        return df1.set_index('pathway').join(df2.set_index('pathway'))
                
    def _find_overlaps(self, rwrDf, pathway_dict):
        """return diction with pathway:geneList"""
        # create result dictionary
        result_dict = {} #pathway:sharedGeneList
        # get ppiGenes
        ppi_gene_list = rwrDf.columns.tolist()
        # find overlaps
        for pathway, geneList in pathway_dict.items():
            shareGene_list = sorted(list(set(geneList) & set(ppi_gene_list)))
            result_dict.update({pathway:shareGene_list})
        return result_dict
               
    def _cal_zscore(self, score, scoreList):
        """return zscore and pvalue by lookup table"""
        if np.std(scoreList) != 0:
            zscore = (score - np.mean(scoreList) ) / np.std(scoreList)
            pvalue = scistat.norm.sf(abs(zscore)) # not pdf
            #print('score={:}, scoreList={:}, zscore={:}, pvalue={:}'.format(
            #       score, scoreList[:10], zscore, pvalue))
        else:
            zscore, pvalue = np.nan, np.nan
        return [zscore, pvalue]

    def _cal_similarity_score(self, rwrDf, geneList):
        """return similarity score by taking average of rwr for given geneList"""
        return rwrDf.loc[geneList].mean()
            
    def _get_pathway_genes(self, gmt):
        """
        Return pathwayStr_geneList_dict

        :param fin: file name to pathway in gmt format
        :return pathway_dict: dictionary of pathway as key, genelist as values
        """
        pathwayStr_geneList_dict = {}
        with open(gmt, 'r') as f:
            for line in f:
                # extract fields
                line = line.strip('\n').split('\t')
                pathway_str = line[0]
                gene_list = line[2:]
                # update to dict
                pathwayStr_geneList_dict.update({pathway_str:gene_list})
        return pathwayStr_geneList_dict

    def _df2dict(self, df):
        """return 1 by N dataframe to dictionary of N keys"""
        return df.to_dict('records')[0] # keys are column names = gene nams


if __name__ == "__main__":
    # timer 
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    start_time = datetime.now()
    rwr_df = 'test.txt' #'/repo4/ytang4/PHD/db/GDSC/processed/GDSC.MUTCNV.STRING.RWR.txt'
    pathway_gmt = '/repo4/ytang4/PHD/db/MSigdb/c2.cp.pid.v7.1.symbols.gmt'
    # initiate
    cell_pathway_df = NetPEA(rwr_df, pathway_gmt, permutation=3, seed=42, n_cpu=5, out_path='./test_netpea/GDSC')
    spend = datetime.strptime(str(datetime.now()), datetimeFormat) - datetime.strptime(str(start_time),datetimeFormat)
    print( '[Finished in {:}]'.format(spend) )

