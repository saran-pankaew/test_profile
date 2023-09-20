import pandas as pd
import numpy as np
from .exp_categorization import *

# Normalization functions
def binarize(df, axis=0):
    bin_df = df.transpose()[[False for n in range(df.shape[1])]].transpose()
    if axis==0:
        for i in range(df.shape[1]):
            gene = df.iloc[:,i]
            g = mixture.GaussianMixture(n_components=2)
            g.fit(np.array(df.iloc[:,i]).reshape(-1, 1))
            if g.means_[0]<g.means_[1]:
                #print(1)
                bin_df[df.columns[i]]=g.predict(np.array(df.iloc[:,i]).reshape(-1, 1))
            else:
                #print(2)
                bin_df[df.columns[i]] = np.abs(g.predict(np.array(df.iloc[:,i]).reshape(-1, 1))-1)

    else:
        bin_df = bin_df.transpose()
        for i in range(df.shape[0]):
            gene = df.iloc[i,:]
            
            g = mixture.GaussianMixture(n_components=2)
            g.fit(np.array(df.iloc[:,1]).reshape(-1, 1))
            g.predict(np.array(df.iloc[:,1]).reshape(-1, 1))
            bin_df[df.index[i]]=g.predict(np.array(df.iloc[:,i]).reshape(-1, 1))
    return bin_df

def inflated0_norm(df):
    if len(df.columns)!=0:
        percent_1 = np.nanpercentile(df, 1, axis = 0)
        percent_99 = np.nanpercentile(df, 99, axis = 0)
        df_norm = (df - percent_1)/(percent_99 - percent_1)
        df_norm[df_norm>1]=1
        df_norm[df_norm<0]=0
    else:
        df_norm=df
    return df_norm

def uni_norm(df):
    df_ = df.copy()
    lambd = np.log(3)/median_abs_deviation(df_, nan_policy = 'omit')
    df_norm = 1/(1+np.exp(-lambd*(df_-df_.median(skipna = True))))
    return df_norm

def transition_rates_calculation(df_rna_count_matrix, dict_nodes_genes=None,\
                                 max_percent = 0.95, diptest_max=0.05, bi_min=1.5, kurtosis_max=1,\
                                 amplification_factor = 100, return_initial_states=True):
    """ This function defines personalized transition rates values for Boolean model simulations with MaBoSS according to
        the workflow proposed in the publication of J. Béal et al. (2018), 
        'Personalization of Logical Models With Multi-Omics Data Allows Clinical Stratification of Patients' 
   
   ______ 
   inputs
    - df_rna_count_matrix (DataFrame): dataframe of RNA expression, where the samples are the lines and the genes are the columns.
    - dict_nodes_genes (dict):  dictionary, where the keys are the nodes of a model and the values the gene names corresponding to this nodes
    
    _______
    outputs
    - transition_rates_up (DataFrame): the transition rates of the activation of the genes for each sample. 
    They are sufficients to personalize the model but if the  transition_rates_down are needed
    the can be obtained by 1/tranisition_rates_up
    """
    df = df_rna_count_matrix.copy()
    if type(dict_nodes_genes)==dict:
        #filter the genes that are in the model
        dict_nodes2 = {node: dict_nodes_genes[node] for node in dict_nodes_genes if dict_nodes_genes[node] in df.columns}
        df = df.loc[:,dict_nodes2.values()]
        df = df.loc[:,~df.columns.duplicated()]


        #rename df with model nodes directly ?
        #inv_map = {v: k for k, v in dict_nodes2.items()}
        #df = df.rename(columns = inv_map)
        
    else:
        print("the dictionary providing the genes corresponding to the nodes of a model has not been defined")
    
    #filter the genes on the percent of cells their values are 0.
    df_filt = discard_percent_0s(df, max_percent=max_percent)
    
    #get the index of the genes that seems to be binarizable
    ind_bin = binarizable(df_filt, diptest_max=diptest_max, bi_min=bi_min, kurtosis_max=kurtosis_max)
    #normalzie these genes and store the result in bin_df
    print("binarization of {} genes started.".format(len(ind_bin)))
    bin_df = binarize(df_filt[ind_bin])
    print("binarization of {} genes done.\n".format(len(ind_bin)))

    #group all the other genes in another df
    no_bin_ind = [ind for ind in df_filt.columns if ind not in ind_bin]
    df_no_bin = df_filt[no_bin_ind]
    
    #get the index of the genes that seems to follow a 0 inflated distribution
    inflated0_ind = inflated0_test(df_no_bin)
    #normalize their values and store it in inflated0_df
    inflated0_df = inflated0_norm(df_no_bin[inflated0_ind])

    #get all the other indexes 
    univar_ind = [ind for ind in df_no_bin.columns if ind not in inflated0_ind]
    #normalize their expressions
    univar_df = uni_norm(df_no_bin[univar_ind])

    print("normalization of {} genes done.".format(len(no_bin_ind)))
    
    total_binnorm_df = pd.concat([bin_df, univar_df, inflated0_df], axis=1)
    #return the binarized/normalized values return total_binnorm_df

    transitions_up = amplification_factor**(2*(np.array(total_binnorm_df)-0.5))
    df_tr_up = pd.DataFrame(transitions_up, index=total_binnorm_df.index, columns=total_binnorm_df.columns)
    if return_initial_states == True:
        return df_tr_up, total_binnorm_df
    else:
        return df_tr_up
    
def data_normalization(df_rna_count_matrix, 
                       dict_gene_node = None,
                       diptest_max=0.05, bi_min=1.5, kurtosis_max=1, max_percent = 0.95):
    
    df_ = df_rna_count_matrix.copy()
    
    # Use function `calculate_statistic`
    criteria_df = calculate_statistic(df_, dict_gene_node = dict_gene_node,
                                      diptest_max = diptest_max,
                                      bi_min = bi_min,
                                      kurtosis_max= kurtosis_max,
                                      max_percent = max_percent)
    criteria_df.reindex(index = df_.columns)
    
    # Start normalization
    ## Inflated0_genes
    inflated0_genes = criteria_df.index[criteria_df.Category=='ZeroInf']
    inflated0_df = inflated0_norm(df_[inflated0_genes])

    ## Unimodel_genes
    unimodal_genes = criteria_df.index[criteria_df.Category=='Unimodal']
    unimodal_df = uni_norm(df_[unimodal_genes])

    ## Inflated0_genes
    bimodal_genes = criteria_df.index[criteria_df.Category=='Bimodal']
    bimodal_df = binarize(df_[bimodal_genes])
    
    # Obtain the results
    df_norm = pd.concat([bimodal_df, unimodal_df, inflated0_df], axis=1)
    
    return df_norm

def calculate_transition_rates(df_norm, amplification_factor=100):
    transitions_up = amplification_factor**(2*(np.array(df_norm)-0.5))

    df_tr_up = pd.DataFrame(transitions_up, index=df_norm.index, columns=df_norm.columns)
    return df_tr_up
