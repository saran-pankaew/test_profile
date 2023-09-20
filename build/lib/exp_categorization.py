""""functions to perform categorization of expression data (transcriptomics + proteomics)"""

import pandas as pd
from sklearn import mixture
from scipy import stats
from  scipy.stats import kurtosis, median_abs_deviation #
import diptest as dptest
import numpy as np

def percent_0s(df, axis=0):
    n_0s = np.sum(df==0, axis=axis)
    return n_0s/df.shape[axis]

def discard_percent_0s(df, max_percent=0.95,axis=0):
    p_0s = percent_0s(df, axis=axis)
    if axis==0:
        df_filt = df.transpose()[p_0s<=max_percent]
        return df_filt.transpose()
    else:
        return df[p_0s<=max_percent]
    
#function combinating the three methods to define binarizable variables used in PROFILE
def binarizable(df, bi_min=None, kurtosis_max=None, diptest_max=None, return_df=False):
    df_ = df.copy()
    if (bi_min==None) & (kurtosis_max==None) & (diptest_max==None):
        print('No filter will be applied since no threshold has been precised.')
        return None
    if bi_min != None:
        df_ = df_.transpose()[bimodality_index(df_)>=bi_min].transpose()
    if kurtosis_max != None:
        df_ = df_.transpose()[kurtosis(df_)<kurtosis_max].transpose()
    if diptest_max!=None:
        df_ = df_.transpose()[diptest(df_)<diptest_max].transpose()
    if return_df==False:
        return df_.columns
    else:
        return df_

#method to calculate the bimodality index for all the variables of a dataframe
def bimodality_index(df, axis=0, return_df = False):
    #don't forget that mixture gives a different result at each run since the process starts from a random state.
    BIs = []
    if axis==0:
        for i in range(df.shape[1]):
            # Fit 2-component Gaussian mixture model
            g = mixture.GaussianMixture(n_components=2, random_state=0)
            gene = df.iloc[:,i]
            gene = gene[gene.notna()]
            g.fit(np.array(gene).reshape(-1, 1))
            
            # Calculate Bimodality index
            sigma = np.sqrt(np.mean(g.covariances_))
            delta = abs(g.means_[0] - g.means_[1])/sigma
            pi = g.weights_[0]
            BI = delta * np.sqrt(pi*(1-pi))
            BIs.append(BI)
    else:
        for i in range(df.shape[0]):        
            g = mixture.GaussianMixture(n_components=2)
            g.fit(np.array(df.iloc[i,:]).reshape(-1, 1))
                  
            sigma = np.sqrt(np.mean(g.covariances_))
            delta = abs(g.means_[0] - g.means_[1])/sigma
            pi = g.weights_[0]
            BI = delta * np.sqrt(pi*(1-pi))
            BIs.append(BI)
    if return_df == True:
        bi_mod_ind = pd.DataFrame(BIs, index = df.columns, columns = ['BI'])
        return bi_mod_ind 
    else:
        return np.array(BIs)

def diptest(df, axis=0):
    dip_pvalue = []
    if axis==0:
        print(df.shape)
        for i in range(df.shape[1]):
            gene = df.iloc[:,i]
        
            # dip_pvalue.append(diptst(gene)[1])
            dip_pvalue.append(dptest.diptest(gene)[1])
        dip_pvalue = pd.DataFrame(dip_pvalue, index=df.columns, columns = ['Dip'])
        return dip_pvalue
    else:
        raise

def inflated0_test(df):
    amplitudes = np.max(df) - np.min(df)
    peaks=[]
    results = []    
    for i in range(df.shape[1]):
        gene = df.iloc[:,i]
        gene = gene[gene.notna()]
        kde1 = stats.gaussian_kde(gene)
        
        x_eval = np.linspace(np.min(gene), np.max(gene), 1000)
        proba = kde1(x_eval)
        
        peaks.append(x_eval[np.argmax(proba)])
        if peaks[-1]<amplitudes[i]/10:
            results.append(True)
        else:
            results.append(False)
    return df.columns[results]

def meanNZ_cal(df):
    meanNZ=[]
    for i in range(df.shape[1]):
        gene = df.iloc[:,i]
        gene=gene[(gene>0)]
        meanNZ.append(gene.mean())
    return meanNZ

# Command functions
def calculate_statistic(df, dict_gene_node = None,
                        diptest_max=0.05, bi_min=1.5, kurtosis_max=1, max_percent = 0.95):
    df_ = df.iloc[:,df.columns.isin(dict_gene_node.values())].copy()
    # Calculate diptest
    dip_ind = diptest(df_)

    # BI
    bi_mod_ind = bimodality_index(df_, return_df = True)
    bi_mod_ind = pd.DataFrame(bi_mod_ind, index = df_.columns, columns = ['BI'])

    # Kurtosis
    kurt_ind = kurtosis(df_)
    kurt_ind = pd.DataFrame(kurt_ind, index = df_.columns, columns = ['Kurtosis'])

    # Drop-out rates
    drop_out_rates = percent_0s(df_)
    drop_out_rates = pd.DataFrame(drop_out_rates, index = df_.columns, columns = ['DropOutRate'])

    # MeanNZ
    meanNZ = meanNZ_cal(df_)
    meanNZ = pd.DataFrame(meanNZ, index = df_.columns, columns = ['MeanNZ'])

    # Density Peak
    peaks = []
    for i in range(df_.shape[1]):
        gene = df_.iloc[:,i]
        mask = ~np.isnan(gene)
        gene = gene[mask]
        kde1 = stats.gaussian_kde(gene, bw_method = 'silverman')
        
        x_eval = np.linspace(np.min(gene), np.max(gene), 1000)
        proba = kde1(x_eval)
        
        peaks.append(x_eval[np.where(proba==max(proba))[0]][-1])
    peaks = pd.DataFrame(peaks, index = df_.columns, columns = ['DenPeak'])

    # Amplitudes
    amplitudes = np.max(df_, axis = 0) - np.min(df_, axis = 0)
    amplitudes = pd.DataFrame(amplitudes, index = df_.columns, columns = ['Amplitude'])

    # Bring it all together
    criteria_df = pd.concat([dip_ind, bi_mod_ind, kurt_ind,
                              drop_out_rates, meanNZ, peaks, amplitudes],
                             axis= 1)
    

    # Add the summarized criteria
    ## get the index of genes that will be discarded
    discard_ind = criteria_df.loc[(criteria_df.DropOutRate>= max_percent),:].index
    ## get the index of genes that are binary distributed
    ### group all the other genes in another df
    no_bin_ind = [ind for ind in criteria_df.index if ind not in discard_ind]
    criteria_df_no = criteria_df.loc[no_bin_ind,:]
    bin_ind = criteria_df_no.loc[(criteria_df_no.BI > bi_min)&(criteria_df_no.Dip<diptest_max)&(criteria_df_no.Kurtosis < kurtosis_max),:].index
    
    ## get the index of genes that are inflated0 distributed
    ### group all the other genes in another df
    no_bin_ind = [ind for ind in criteria_df_no.index if ind not in bin_ind]
    criteria_df_no = criteria_df_no.loc[no_bin_ind,:]
    inflated0_ind = criteria_df_no.loc[(criteria_df_no.DenPeak<(criteria_df_no.Amplitude/10)),:].index

    ## get the index of genes that are unimodally distributed
    univar_ind = [ind for ind in criteria_df_no.index if ind not in inflated0_ind]

    ## Summarize in array
    criteria_df['Category'] = list(range(len(criteria_df)))
    criteria_df.loc[discard_ind,'Category'] = 'Discarded'
    criteria_df.loc[bin_ind,'Category'] = 'Bimodal'
    criteria_df.loc[inflated0_ind,'Category'] = 'ZeroInf'
    criteria_df.loc[univar_ind,'Category'] = 'Unimodal'

    return criteria_df