import os
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import maboss

# Define the personalize_model function
def personalize_model(cell, cellname, model, dict_mut, dict_tr, dir, sample_count, previous):
    personalized_model = model.copy()
    nodes = model.network.names
    for node in nodes:
        # Assign node mutation
        if node in dict_mut:
            gene = dict_mut[node]
            if gene in cell['mutations'].keys():
                val_mut = cell['mutations'][gene]
                personalized_model.mutate(node, val_mut)
        # Assign transition rates
        if node in dict_tr:
            gene = dict_tr[node]
            if gene in cell['transition_rates_up'].keys():
                personalized_model.param['$u_'+node] = cell['transition_rates_up'][gene]
                personalized_model.param['$d_'+node] = 1/cell['transition_rates_up'][gene]
        # Assign initial states
        if len(cell['initial_states'])!=0:
            for node in cell['initial_states']:
                personalized_model.network.set_istate(node, cell['initial_states'][node], warnings= False)
                personalized_model.update_parameters(sample_count = sample_count)

    # Write the personal model in the sc_tmp directory
    personalized_model_path = dir+'/'+cellname
    personalized_model.print_bnd(open(personalized_model_path+'.bnd', 'w'))
    personalized_model.print_cfg(open(personalized_model_path+'.cfg', 'w'))
    
    # For multiprocessing
    previous.append(cellname)

# Define the run simulation function
def run_cellmodel_simulation(cell, cellname, dir, previous):
    # Run and save the simulation results
    model_res = cell.run()
    model_res.save(dir+'/'+cellname+'_res')
    # For multiprocessing
    previous.append(cellname)

class CellEnsemble:
    def __init__(self, df_mutations, df_transition_rates, df_init_states, dict_gene_nodes, dict_strict_gene_nodes, project_name, df_groups):
        self.cells = {}
        self.mutations = df_mutations
        self.tr_rates = df_transition_rates
        self.init_cond = df_init_states
        self.dict_mut = dict_gene_nodes
        self.dict_tr = dict_strict_gene_nodes
        self.groups = df_groups
        # For project name -> If not defined - use date as project name
        if type(project_name) is str:
            self.name = project_name
        else:
            now = datetime.now()
            now = now.strftime("%Y_%m_%d_%Hh%M")
            self.name = print(now)

        # Transition rates - create a dictionary object for transition rates
        dic_tr = df_transition_rates.transpose().to_dict()
        
        # Initial conditions- create a dictionary object for initial conditions
        dic_init_states = df_init_states.transpose().to_dict()
        f_init_states = {}

        # For loop to create init cond for each cell
        for cell in dic_tr.keys():
            f_init_states[cell]={}
            for gene in dic_init_states[cell]:
                for node in dict_strict_gene_nodes:
                    if gene in dict_strict_gene_nodes[node]:
                        f_init_states[cell][node] = {0:1-dic_init_states[cell][gene], 1:dic_init_states[cell][gene]}                
        dic_init_states=f_init_states

        # For loop to create dictionary of parameters for each cells
        for cell in tqdm(dic_tr.keys()):

            # Get cellline name for each cell
            cellline = self.groups[cell]
            
            # Create mutation profile
            dic_mut = {k:v for k,v in df_mutations[cellline].to_dict().items() if v in ('ON','OFF')}
            
            # Create the dictionary of cells
            self.cells[cell] = {"name":cell,
                                "mutations":dic_mut,
                                "transition_rates_up":dic_tr[cell],
                                "dict_gene_nodes":dict_gene_nodes,
                                'initial_states':dic_init_states[cell],
                                'dict_strict_gene_nodes':dict_strict_gene_nodes
                                }

    def __repr__(self):
        return 'conditions : ' + str(list(self.model.keys())) + '\ngroups : ' + str(self.groups.cat.categories)

    def personalize_cellmodel(self, model, num_processes, reload_model = True, simulation_sample=1000):
        # Create base model condition
        self.model = {'base':{}}

        # Arg num_processes
        if num_processes == None:
            num_processes = 1
        elif type(num_processes) == int:
            num_processes = num_processes

        # Multiprocessing arg
        manager=mp.Manager()
        previous=manager.list()
        processes=[]

        # Set parameters
        self.core_model = model.copy()
        cells = list(self.cells.keys())

        # Create the sc_tmp directory
        base_folder = os.getcwd()
        if not os.path.exists(base_folder+'/sc_tmp/'):
            os.makedirs(base_folder+'/sc_tmp/')

        # Set the sample_count parameter
        if type(simulation_sample) == int:
            simulation_sample = simulation_sample
             
        # Create the project directory
        project_folder = base_folder + '/sc_tmp/'+self.name
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)
        else:
            now = datetime.now()
            now = now.strftime("%Y_%m_%d_%Hh%M")
            project_folder = project_folder + '_' + now
            os.makedirs(project_folder)
        self.project_dir = project_folder + '/'

        # Create base condition folder
        condition_dir = self.project_dir + 'base/'
        os.makedirs(condition_dir)

        # For loop for every group in the dataset
        print('Parameterizing the model')
        for group in tqdm(self.groups.cat.categories):
            
            # Create group directory
            group_dir = condition_dir + group
            os.makedirs(group_dir)
            #print('Parameterizing model from group : ' + group + '\n')
            cells = self.groups[self.groups == group].index

            # Assign the sample_count for each cell group
            group_sample_count = len(cells)*simulation_sample
            
            # For loop for every cells in the group
            for i in range(len(cells)):
                cell = cells[i]
                while len(previous)<i-(num_processes-1):
                    time.sleep(1)
                p = mp.Process(target = personalize_model, 
                               args = (self.cells[cell], cell, self.core_model, self.dict_mut, self.dict_tr, group_dir, group_sample_count,previous))
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
        
        # Reload the model
        if reload_model == True :
            print('loading model results...')
            for cell in tqdm(cells):
                self.model['base'][cell] = maboss.load(bnd_filename=group_dir+'/'+cell+'.bnd',
                                                       cfg_filenames=group_dir+'/'+cell+'.cfg')
                
    def load_personalized_model(self, dir):
        # Create base model condition
        self.model = {'base':{}}

        # Add the folder which 
        self.project_dir = dir+'/'
        cells = list(self.cells.keys())
        
        # Reload the model
        print('loading model results...')
        for cell in tqdm(cells):
            self.model['base'][cell] = maboss.load(bnd_filename=self.project_dir+cell+'.bnd',
                                                  cfg_filenames=self.project_dir+cell+'.cfg')

    def add_conditions(self, nodes, condition, condition_name, num_processes, simulation_sample=1000):
        # Arg num_processes
        if num_processes == None:
            num_processes = 1
        elif type(num_processes) == int:
            num_processes = num_processes

        # Multiprocessing arg
        manager=mp.Manager()
        previous=manager.list()
        processes=[]

        # Create base condition folder
        condition_dir = self.project_dir + condition_name
        os.makedirs(condition_dir)

        # Mutated the core model
        self.mutated_model = maboss.copy_and_mutate(self.core_model, nodes=nodes, mut = condition)

        # For loop for every group in the dataset
        print('Parameterizing the model')
        for group in tqdm(self.groups.cat.categories):
            # Create group directories
            group_dir = condition_dir+'/'+group
            os.makedirs(group_dir)
            cells = self.groups[self.groups == group].index

            # Assign the sample_count for each cell group
            group_sample_count = len(cells)*simulation_sample

            # For loop for every cells in the group
            for i in range(len(cells)):
                cell = cells[i]
                while len(previous)<i-(num_processes-1):
                    time.sleep(1)
                p = mp.Process(target = personalize_model, 
                               args = (self.cells[cell], cell, self.mutated_model, self.dict_mut, self.dict_tr, group_dir, group_sample_count,previous))
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

    def run_simulation(self, conditions='all', redo = False, num_processes = None):
        # Arg conditions
        if conditions == 'all':
            conditions = list(self.model.keys())
        elif type(conditions) not in [list, tuple]:
            conditions = [conditions]
        
        # Arg num_processes
        if num_processes == None:
            num_processes = 1
        elif type(num_processes) == int:
            num_processes = num_processes

        # Multiprocessing arg
        manager=mp.Manager()
        previous=manager.list()
        processes=[]
        
        # Create temporary results folder
        self.result_dir = self.project_dir + 'tmp_res/'

        # Get cells list
        cells = list(self.cells.keys())

        # For loop to run simulation for each condtion
        print('Compute simulations')
        for condition in conditions:
            print('\tComputing simulations for condition : ' + condition)
            condition_dir = self.result_dir + condition
            if not os.path.exists(condition_dir):
                os.makedirs(condition_dir)

            # For loop to run simulation for each cell
            for i in tqdm(range(len(cells))):
                cell = list(self.model[condition].keys())[i]
                if redo==False:
                    try:
                        self.model[condition][cell].result
                        print('the simulation {}|{} will not be re-computed'.format(cell, condition))
                        continue
                    except:
                        pass
                while len(previous)<i-(num_processes-1):
                    time.sleep(1)
                p = mp.Process(target = run_cellmodel_simulation, 
                               args = (self.model[condition][cell], cell, condition_dir, previous))
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

    def save_models(self, conditions='all'):
        # Create and define base directory for models
        ## Create scMODELs folder
        base_folder = os.getcwd()
        if not os.path.exists(base_folder+'/scMODELS'):
            os.makedirs(base_folder+'/scMODELS')
        ## Create project folder
        project_folder = base_folder+'/scMODELS/'+self.name
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)
        else:
            now = datetime.now()
            now = now.strftime("%Y_%m_%d_%Hh%M")
            project_folder = project_folder + '_' + now
            os.makedirs(project_folder)
        self.project_dir = project_folder

        # Arg conditions
        if conditions == 'all':
            conditions = list(self.model.keys())
        elif type(conditions) not in [list, tuple]:
            conditions = [conditions]

        # For loop to run each conditions
        for condition in conditions:
            condition_dir = project_folder+'/'+condition
            os.makedirs(condition_dir)
            print('Saving models from condition:' + condition + '\n')
            # For loop for each cellline/samples
            for group in self.groups.cat.categories:
                group_dir = condition_dir+'/'+group
                os.makedirs(group_dir)
                cells = self.groups[self.groups==group].index
                print('\t Saving models from group : ' + group)
                for cell in tqdm(cells):
                    path_bnd = group_dir + '/' + cell + '.bnd'
                    path_cfg = group_dir + '/' + cell + '.cfg'
                    self.model[condition][cell].print_bnd(open(path_bnd,'w'))
                    self.model[condition][cell].print_cfg(open(path_cfg,'w'))

    def get_nodeprob_matrix(self, cells = 'all', conditions = 'all', fill_na = True):
        # Arg cell
        if cells == 'all':
            cells = list(self.results['base'].keys())
        elif type(cells) in [list, tuple]:
            cells = [cells]
        
        # Arg conditions
        if conditions == 'all':
            conditions = list(self.results.keys())
        elif type(conditions) not in [list, tuple]:
            conditions = [conditions]

        # for loop to obtain node_prob traj
        probtraj = {}
        for condition in conditions:
            probtraj_mtx=pd.DataFrame()
            for cell in cells:
                try:
                    probtraj_cell=self.results[condition][cell].get_last_nodes_probtraj()
                    probtraj_mtx = pd.concat([probtraj_mtx,probtraj_cell], ignore_index=True)
                except:
                    print('the simulation {}|{} seems not having been computed'.format(cells, condition))
            probtraj_mtx.index = cells
            if fill_na == True:
                probtraj_mtx = probtraj_mtx.fillna(0)
            probtraj[condition] = probtraj_mtx
        return probtraj
    
    def get_stateprob_matrix(self, cells = 'all', conditions = 'all', fill_na = True):
        # Arg cell
        if cells == 'all':
            cells = list(self.results['base'].keys())
        elif type(cells) in [list, tuple]:
            cells = [cells]
        
        # Arg conditions
        if conditions == 'all':
            conditions = list(self.results.keys())
        elif type(conditions) not in [list, tuple]:
            conditions = [conditions]
        
        # for loop to obtain state_prob traj
        probtraj = {}
        for condition in conditions:
            probtraj_mtx=pd.DataFrame()
            for cell in cells:
                try:
                    probtraj_cell=self.results[condition][cell].get_last_states_probtraj()
                    probtraj_mtx = pd.concat([probtraj_mtx,probtraj_cell], ignore_index=True)
                except:
                    print('the simulation {}|{} seems not having been computed'.format(cells, condition))
            probtraj_mtx.index = cells
            if fill_na == True:
                probtraj_mtx = probtraj_mtx.fillna(0)
            probtraj[condition] = probtraj_mtx
        return probtraj