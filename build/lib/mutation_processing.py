from tqdm import tqdm
import pandas as pd
import re
import numpy as np

pd.options.mode.chained_assignment = None

def load_polyphen(work_dir):
    # Set base folder directory
    base_folder = work_dir
    
    # Load Polyphen database
    polyphen = pd.read_table(base_folder+"/pph2-full_all.txt", 
                         sep = '\t', 
                         header = 0,
                         low_memory=False)
    polyphen.columns = polyphen.columns.str.strip()
    polyphen.rename(columns= {'#o_acc':"Gene"}, inplace = True)
    polyphen['Label'] = polyphen['Gene'].str.strip() + '_' + polyphen['o_aa1'].str.strip() + polyphen['o_pos'].astype('str') + polyphen['o_aa2'].str.strip()
    polyphen['Polyphen'] = polyphen['prediction'].str.strip()
    polyphen = polyphen[['Label','Polyphen']]
    return polyphen

def load_oncokb(work_dir):
    # Set base folder directory
    base_folder = work_dir
    # Load OncoKB database
    OncoKB = pd.read_table(base_folder+"/allAnnotatedVariants.txt", 
                       sep = '\t', 
                       encoding = 'latin1')
    OncoKB.columns = OncoKB.columns.str.strip()
    OncoKB['Label'] = OncoKB['Gene'].str.strip() + '_' + OncoKB['Alteration'].str.strip()
    bin_label = {'Gain-of-function':1,
            'Likely Gain-of-function':1,
            'Loss-of-function':0,
            'Likely Loss-of-function':0,
            'Inconclusive':np.nan,
            'Likely Neutral':np.nan,
            'Switch-of-function':np.nan,
            'Neutral':np.nan,
            'Likely Switch-of-function':np.nan}
    OncoKB['BIN'] = OncoKB['Mutation Effect'].map(bin_label) 
    oncokb_label = OncoKB[['BIN']].set_index(OncoKB['Label']).to_dict()
    oncokb_label = oncokb_label['BIN']
    return oncokb_label

def load_tsg(work_dir):
    # Set base folder directory
    base_folder=work_dir
    # Load 2020+ database
    OncoTSG = pd.read_table(base_folder+"/2020_pancancer.csv",
                        sep=';', 
                        skiprows=1)
    OncoTSG = OncoTSG[['gene','oncogene q-value', 'tsg q-value']]
    return OncoTSG

def mutation_assignment(mutation_data, database_dir):
    # Load all databases
    ## 2020+ database
    polyphen = load_polyphen(database_dir)
    OncoTSG = load_tsg(database_dir)
    oncogenes = list(OncoTSG['gene'][OncoTSG['oncogene q-value']<=0.1])
    tsg = list(OncoTSG['gene'][OncoTSG['tsg q-value']<=0.1])
    ### Create polyphen dictionary for oncogenes
    polyphen['gene_symbol'] = polyphen['Label'].str.split("_", n = 1, expand = True)[0]
    polyphen_oncogenes = polyphen[polyphen['gene_symbol'].isin(oncogenes)]
    polyphen_oncogenes['BIN'] = polyphen_oncogenes['Polyphen'].map({'probably damaging':1,
                                                                    'possibly damaging':1,
                                                                    'benign':np.nan,
                                                                    'unknown':np.nan})
    dict_oncogenes = polyphen_oncogenes[['BIN']].set_index(polyphen_oncogenes['Label']).to_dict()['BIN']
    ### Create polyphen dictionary for tsg
    polyphen_tsg = polyphen[polyphen['gene_symbol'].isin(tsg)]
    polyphen_tsg['BIN'] = polyphen_tsg['Polyphen'].map({'probably damaging':0,
                                                        'possibly damaging':0,
                                                        'benign':np.nan,
                                                        'unknown':np.nan})
    dict_tsg = polyphen_tsg[['BIN']].set_index(polyphen_tsg['Label']).to_dict()['BIN']

    ## OncoKB database
    oncokb_label = load_oncokb(database_dir)
    

    # Assign mutation profile
    ## Create the Label column in the mutation profile matrix
    mutation = mutation_data
    mutation['Label'] = mutation['gene_symbol'] + '_' + mutation['protein_mutation'].str[2:]
    mutation['Method'] = np.nan
    mutation['BIN'] = np.nan

    ## Check OncoKB database
    print('Check OncoKB mutation database')
    for line in tqdm(range(len(mutation))):
        if mutation['Label'].iloc[line] in oncokb_label.keys():
            k = mutation['Label'].iloc[line]
            mutation['Method'].iloc[line] = 'OncoKB'
            mutation['BIN'].iloc[line] = oncokb_label[k]

    ## Check Inactivating mutation
    print('Check Inactivating mutation')
    for line in tqdm(range(len(mutation))):
        if (mutation['Method'].iloc[line]!= "OncoKB"):
            if re.search("p\\.\\*[0-9]*[A-Z]|p\\.[A-Z][0-9]*\\*|p\\.[A-Z][0-9]*fs\\*", mutation['protein_mutation'].iloc[line]):
                mutation['Method'].iloc[line] = 'Inactivating'
                mutation['BIN'].iloc[line] = 0
    
    ## Check 2020+ database
    print('Check 2020+ mutation database')
    for line in tqdm(range(len(mutation))):
        if ((mutation['Method'].iloc[line]!= "OncoKB")&(mutation['Method'].iloc[line]!= "Inactivating")&(mutation['gene_symbol'].iloc[line] in oncogenes+tsg)):
            mutation['Method'].iloc[line] = '2020+'
            k = mutation['Label'].iloc[line]
            if mutation['Label'].iloc[line] in dict_oncogenes:
                mutation['BIN'].iloc[line] = dict_oncogenes[k]
            if mutation['Label'].iloc[line] in dict_tsg:
                mutation['BIN'].iloc[line] = dict_tsg[k]

    ## Write the remaining categories           
    mutation['Method'] = mutation['Method'].replace(np.nan, 'Not Processed')
    #mutation['BIN'] = mutation['BIN'].replace(np.nan, 'Not Assigned')

    return mutation


def mutation_fusion(mutation_data):
    # Obtain the unique gene list from all model_name
    mutation = mutation_data[['model_name','gene_symbol','BIN']]
    mutation['model_name'] = mutation['model_name'].astype('category')
    gene_name = {}
    for i in list(mutation['model_name'].cat.categories):
        data = mutation[mutation['model_name'] == i]
        data = data[data.BIN.notna()]['gene_symbol']
        gene_name[i] = list(data)
    gene_name = list(set().union(*gene_name.values()))
    
    # Subset the matrix
    mutation = mutation[mutation['gene_symbol'].isin(gene_name)]

    # Rearrange dataframe
    new_mutation = pd.pivot_table(mutation, values = 'BIN', index='gene_symbol', columns = 'model_name')

    return new_mutation