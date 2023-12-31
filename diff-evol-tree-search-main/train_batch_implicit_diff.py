from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import os
import sys

import optax
import jaxopt
from jaxopt import OptaxSolver
from jaxopt import linear_solve
from jax.tree_util import Partial

import wandb
import pydot
import warnings
import argparse
import functools
from typing import Dict
import copy

import pickle
import dendropy
import ete3
from Bio import Phylo

from matplotlib import rc
import matplotlib as plt
import plotly.express as px
rc('animation', html='jshtml')

import jax
## use cpu for the time being, let's see whether user requested a gpu later.
jax.config.update("jax_default_device", jax.devices("cpu")[0]) 
import jax.nn as nn
from jax import vmap
import jax.numpy as jnp
from jax.lib import xla_bridge
from jaxtyping import Array, Float

import numpy as np

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

from modules.vis_utils import *
from modules.tree_func import *
from modules.gt_tree_gen import * # COMMENT IF STARTING WITH ALEATORY SEQUENCE
from modules.sankoff import *
from arg_parser_v2 import *
#from aleatory_initial_seq_no_fitness import * UNCOMMENT IF STARTING ALEATORY SEQUENCE


## Parse Command Line Arguments and perform checks
args = vars(parse_args_v2())
args = sanity_check(args)

#### Define Sequence length and number of leaves
seq_length  = int(args['seq_length']) if args['seq_length']!=None else 20
n_leaves    = int(args['leaves']) if args['leaves']!=None else 4
n_ancestors = n_leaves - 1
n_all       = n_leaves + n_ancestors
n_mutations = int(args['mutations']) if args['mutations']!=None else 3
n_letters   = int(args['letters']) if args['letters']!=None else 20

#CREATING FOLDER WHERE SAVING SIMULATIONS

folder_name="%ileaves_seql%i_%iletters_%imuts"% (n_leaves, seq_length, n_letters, n_mutations)
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
    print(f"Created folder '{folder_name}'.")
else:
    print(f"Folder '{folder_name}' already exists.")
    
folder_name_images=folder_name+"/images"
if not os.path.exists(folder_name_images):
    os.mkdir(folder_name_images)
    print(f"Created folder '{folder_name}'.")
else:
    print(f"Folder '{folder_name}' already exists.")
    
###useful variable 
    
best_trees=[]
gt_trees=[]
gt_sequences=[]
best_sequences=[]
gt_nb_binary=0
gt_nb_non_binary=0
bt_nb_binary=0
bt_nb_non_binary=0

quant_numbers_dict={
                    'gt_cost':[],
                    'sankoff_cost':[],
                    'best_cost':[],
                    'gt_b1':[],
                    'best_tree_b1':[],
                    'gt_sackin':[],
                    'best_tree_sackin':[],
                    'rf_gt_best_tree':[],
                    'gt_colless':[],
                    'best_tree_colless':[]
        }
    
for n_simul in range(1,101):

    def generate_vmap_keys(seq_params):
        vmap_keys = {}
        
        for key in seq_params.keys():
            vmap_keys[key] = 0
            
        return vmap_keys
    
    @jit
    def get_one_tree_and_seq(tree_params, seq_params, pos):
        new_params = {}
        ## extract correct tree
        new_params['t'] = tree_params['t'][pos]
        
        for i in range(0, len(seq_params.keys())):
            new_params[str(i)] = seq_params[str(i)][pos]
        
        return new_params
    
    
    def inner_objective(seq_params, tree_params, data):
        seqs, metadata, temp, epoch = data
        return compute_loss_optimized(tree_params, seq_params, seqs, metadata, temp, epoch)
    
    
    def inner_loop_solver(seq_params, tree_params, data):
        seqs, metadata, temp, epoch = data
        
        inner_opt_state = seq_optimizer.init_state(seq_params, tree_params, data)
    
        for i in range(0, seq_optimizer.maxiter):
            seq_params, inner_opt_state = jitted_seq_update(seq_params, inner_opt_state, tree_params, data)
    
        return seq_params
    
    
    def outer_objective(tree_params, seq_params, data):
        seqs, metadata, temp, epoch = data
        seq_params = inner_loop_solver(seq_params, tree_params, data)
        return compute_loss_optimized(tree_params, seq_params, seqs, metadata, temp, epoch), seq_params
    
    
    #useful functions
    def rootidx(adj_matrix):
        for r in range(len(adj_matrix)):
            if adj_matrix[r][r] == 1:
                root_index=r
                return root_index
    
    def clading(tree,matrix,indices,done_indices):
        
        new_roots=[]
        
        #we iterate through the "roots", starting from the older ancester and updating it after
        for r in indices:
            
            #the loc is to access the leaf(ves) of a root, we assume to start with one leave
            #and at each new one we assume there is another one (the later: append(loc[-1]+1))
            loc=[0]
            
            #iterating through upper triang. adjacency matrix
            for i in range(len(matrix)):
                
                for j in range(len(matrix)):
                    
                    #if connected to root, and not diagonal and we didn't already went through the row
                    if (matrix[i][j]==1) and (r==j) and (i!=j) and (i not in done_indices):
                        next(tree.find_clades(str(r))).split(n=1)
                        next(tree.find_clades(str(r)))[loc[-1]].name=str(i)
                        
                        #this is to keep track of the roots we have already done
                        done_indices.append(r)
                        
                        #this is now the new root(s) from which we will recall the function
                        new_roots.append(i)
                        
                        loc.append(loc[-1]+1)
                        #we stop when we have all the leaves (which is length of matrix)
                        if (len(tree.get_terminals())+len(tree.get_nonterminals())==len(matrix)):
                            return
        clading(tree,matrix,new_roots,done_indices)
    
    
    
    def adjmat_to_treeclass(matrix):
        
        #creating the tree and setting the root (only place where 1 on diagonal)
        root=Phylo.BaseTree.Clade(1.0, name=str(rootidx(matrix)))
        tree=Phylo.BaseTree.Tree(root, rooted=True)
        
        #we already keep track of the root we went through, starting with initial one
        done_indices=[rootidx(matrix)]
        
        clading(tree,matrix,[rootidx(matrix)],done_indices)
        
        return tree
    #
    
    
    
    ## Parse Command Line Arguments and perform checks
    args = vars(parse_args_v2())
    args = sanity_check(args)
    
    #### Define Sequence length and number of leaves
    seq_length  = int(args['seq_length']) if args['seq_length']!=None else 20
    n_leaves    = int(args['leaves']) if args['leaves']!=None else 4
    n_ancestors = n_leaves - 1
    n_all       = n_leaves + n_ancestors
    n_mutations = int(args['mutations']) if args['mutations']!=None else 3
    n_letters   = int(args['letters']) if args['letters']!=None else 20
    
    args['tree_loss_schedule'] = eval(args['tree_loss_schedule']) if args['tree_loss_schedule']!=None else [0,0.01,100,5]
     
    metadata = {
        'n_all' : n_all,
        'n_leaves' : n_leaves, 
        'n_ancestors' : n_ancestors,
        'seq_length' : seq_length,
        'n_letters' : n_letters,
        'n_mutations' : n_mutations,
        'args': args,
        'exp_name' : f"l={n_leaves}, m={n_mutations}, s={seq_length}, fs={args['fix_seqs']}, ft={args['fix_tree']}" ,
        'seed' : int(args['seed']) if args['seed']!=None else np.random.randint(0,1000),
        'seq_temp': 0.5,
        'lr': args['learning_rate'],
        'lr_seq' : args['learning_rate_seq'] if args['learning_rate_seq']!=None else args['learning_rate']*10,
        'epochs': args['epochs'],
        'tLs': args['tree_loss_schedule'],
        'init_count' : args['init_count'] if args['init_count']!=None else 1
        }
    
    
    
    
    if(args['log_wandb']):
        wandb.login(key = os.environ.get('WANDB_API_KEY_so'))
        wandb.init(project=args['project'], name = args['notes'] + metadata['exp_name'], entity="jefftamburi", config = metadata, tags=["bi-level", args['tags']], notes = args['notes'])
    
    if(args['gpu']!=None):
        print_critical_info(f"Utilizing gpu -> {args['gpu']} \n")
        jax.config.update("jax_default_device", jax.devices("gpu")[args['gpu']])
    else:
        if(xla_bridge.get_backend().platform == "gpu"):
            print_critical_info("There's a gpu available, but you didn't specify to use it 😏. So using cpu instead 🤷🏻‍♂️. \n")    
            print(f"Available GPUs: {jax.devices('gpu')}")
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
    
    print(pretty_print_dict(metadata))
    
    ## JAX Doesn't like some data types when jitting
    metadata['exp_name'] = None
    metadata['args']['notes'] = None
    metadata['notes'] = None
    metadata['tags']  = None
    metadata['project'] = None
    args['notes'] = None
    args['tags'] = None
    args['project'] = None


#### Generate a random sequence of 0s and 1s
    seed=np.random.randint(0,1000)
    key = jax.random.PRNGKey(seed)
    offset = 10
    
    #### Generate a base tree
    base_tree = jnp.zeros((n_all, n_all))
    
    sm = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters']).astype(jnp.float64)
    
    if(args['groundtruth']):
        seqs, gt_seqs, tree, gt_tree_copy = generate_groundtruth(metadata, seed)
        
        
        # print(
        #     "type of seqs:", seqs.dtype,'\n'
        #     "shape of seqs:", seqs.shape,'\n','\n'
        #     "type of gt_seqs:", gt_seqs.dtype,'\n'
        #     "shape of gt_seqs:", gt_seqs.shape,'\n','\n'
        #     "type of tree:", tree.dtype,'\n'
        #     "shape of tree:", tree.shape,'\n'
        #       )
    
        # print(
        #     seqs,'\n','\n',
        #     gt_seqs,'\n','\n',
        #     tree
        #     )
    
        seqs    = jax.nn.one_hot(seqs, n_letters).astype(jnp.float64)
        gt_seqs = jax.nn.one_hot(gt_seqs, n_letters).astype(jnp.float64)
        
        print(n_simul,"/100")
        
        #if we don't want sequences to change, set seqs to gt_seqs
        if(args['fix_seqs'] or args['initialize_seq']):
            seqs = gt_seqs
    
        #if we don't want tree to change, set base_tree to tree
        if(args['fix_tree']):
            base_tree = tree
            
        if(not(args['fix_seqs'])):
            ## since we know the gt tree, let's get the real ancestors using sankoff algorithm! 🎉 
            ## This will help us to see whether there's a better groundtruth ancestors for this tree
            cost_mat = (jnp.ones((n_letters,n_letters)) - jnp.eye(n_letters)).astype(jnp.float64)
            
            print_critical_info("running sankoff on groundtruth tree\n")
            _, _, sankoff_cost = run_sankoff(tree, cost_mat, jnp.argmax(seqs, axis = 2), metadata)
            print_success_info("done running sankoff on groundtruth tree. optimal cost = %d\n" % sankoff_cost)
            
            
            if(args['log_wandb']):
                wandb.log(
                    {
                        #'sankoff_seqs': wandb.data_types.Plotly(px.imshow(sankoff_seqs, text_auto=True)),
                        'sankoff_cost': sankoff_cost,
                        #'cost_for_sankoff_seqs' : compute_cost(jax.nn.one_hot(sankoff_seqs, n_letters).astype(jnp.float64), tree, metadata, surrogate_loss = False)
                    })
                
        if(args['shuffle_seqs']):        
            shuffled_leaves = jax.random.permutation(key, seqs[0:n_leaves], independent=False) 
            seqs = seqs.at[0:n_leaves].set(shuffled_leaves)
            
            shuffled_ancestors = jax.random.permutation(key, seqs[n_leaves:-1], independent=False) 
            seqs = seqs.at[n_leaves:-1].set(shuffled_ancestors)
    
        gt_tree = show_graph_with_labels(tree, n_leaves, True)
        gt_seqs_plot = px.imshow(jnp.argmax(gt_seqs, axis = 2), text_auto=True)
    
        
        gt_cost           = compute_cost(gt_seqs, tree, sm)
        gt_cost_surrogate = compute_surrogate_cost(gt_seqs, tree)
        
        gt_cost_copy=gt_cost.copy()
        
        gt_tree_force_loss = enforce_graph(tree, 10, metadata)
    
        if(args['log_wandb']):
            wandb.log({
                "Groundtruth Tree" : wandb.Image(gt_tree),
                "Groundtruth Seq" : wandb.data_types.Plotly(gt_seqs_plot),
    
                "Groundtruth Tree Force Loss" : gt_tree_force_loss,
                "Groundtruth Traversal Cost (surrogate)" : gt_cost_surrogate,
                "Groundtruth Traversal Cost" : gt_cost,
                "Groundtruth Total Loss" : gt_tree_force_loss + gt_cost
                })
    
    
    #### Define parameters (t and the ancestor sequences)
    initializer = jax.nn.initializers.kaiming_normal()
    
    
    tree_params : Dict[str, Array] = {
        't': initializer(key + offset, (metadata['init_count'], n_all - 1,n_ancestors), jnp.float64)
        }
    
    seq_params : Dict[str, Array] = {}
    
    
    ## Manually override the parameters such that the updated tree is the same as the base tree
    if(args['initialize_tree']):
        print_critical_info("Initializing tree using groundtruth tree \n")
        tree_params['t'] = tree[0:-1,n_leaves:]*100
        
    #### Add the ancestor sequences to the parameters
    for i in range(0, n_ancestors):
        
        if(args['initialize_seq']):
            if(n_leaves > 1024):
                raise NotImplementedError("Sankoff backtracking not implemented for n_leaves > 1024 due to execution time")
            else:
                print_critical_info("Initializing sequences using sankoff ancestors \n")
                seq_params[str(i)] = jax.nn.one_hot(sankoff_seqs[n_leaves + i], n_letters).astype(jnp.float64)*100
        else:
            seq_params[str(i)] = initializer(key+i+offset, (metadata['init_count'], seq_length, n_letters), jnp.float64)
            #jax.random.normal(key+i+offset, (seq_length, n_letters)).astype(jnp.float64)
        
    #### Initialize the optimizer
    
    copy_seq_params = seq_params.copy()
    #eps_root = 1e-8, eps = 1e-8
    
    seq_optimizer = OptaxSolver(opt = optax.adam(metadata['lr_seq'], eps_root = 1e-16), fun = inner_objective, maxiter  = args['alternate_interval'], implicit_diff = True) 
    jitted_seq_update = jax.jit(seq_optimizer.update)
    
    tree_optimizer = OptaxSolver(opt = optax.adam(metadata['lr'], eps_root = 1e-16), fun = outer_objective, has_aux = True) #, implicit_diff = True)
    
    vmap_tree_init = jax.vmap(tree_optimizer.init_state, (0, 0, None),0)
    tree_opt_state = vmap_tree_init(tree_params, seq_params, [seqs, metadata, metadata['tLs'][0], 0])
    jitted_tree_update = jit(vmap(tree_optimizer.update, (0, 0, 0, None),0))
    
    vmap_keys = generate_vmap_keys(seq_params)
    vmap_compute_detailed_loss_optimized =  jit(vmap(compute_detailed_loss_optimized, ({'t':0}, vmap_keys, None, None, None, None, None),0))
    
    
    fixed_dummy_pos = 0 
    
    params = get_one_tree_and_seq(tree_params, seq_params, fixed_dummy_pos)
    
    fig2 = show_graph_with_labels(discretize_tree_topology(update_tree(params), n_all),n_leaves, True)
    
    compute_loss(params, seqs, base_tree, metadata, metadata['tLs'][0])
    
    
    best_ans = 1e9
    best_seq = None
    best_tree = None
    
    pos = 0
    #### Training loop
    for _ in range(metadata['epochs']):
    
        if(_%200==0):
            ###~ Get the current discretized tree
            if(args['fix_tree']): 
                t_ = base_tree
            else:
                t_ = update_tree(params, _, metadata['tLs'][0])
            t_d           = discretize_tree_topology(t_,n_all)
            tree_at_epoch        = show_graph_with_labels(t_d, n_leaves, True)
            tree_matrix_at_epoch = px.imshow(t_, text_auto=True)
    
            ###~ Get the current sequences as a plot
            if(args['fix_seqs']):
                seqs_ = seqs
            else:
                seqs_ = update_seq(params, seqs, metadata['seq_temp'])
            new_seq = px.imshow(jnp.argmax(seqs_, axis = 2), text_auto=True)
            
        
        #=> Update the parameters
        tree_params, tree_opt_state = jitted_tree_update(tree_params, tree_opt_state, tree_opt_state.aux, [seqs, metadata, metadata['tLs'][0],_])
        seq_params = tree_opt_state.aux
    
        cost, cost_surrogate, tree_force_loss, loss = vmap_compute_detailed_loss_optimized(tree_params, seq_params, seqs, metadata, metadata['tLs'][0], sm,  _)
        pos = jnp.argmin(cost)
        
        params = get_one_tree_and_seq(tree_params, seq_params, pos)
        
        if(cost.min() < best_ans):
            if(_%20==0):
                print_success_info("Found a better tree at epoch %d with cost %f from tree %d. (delta at epoch = %d) \n" % (_, cost.min(), pos, cost.max()-cost.min()))
            best_ans = cost.min()
            
            t_ = update_tree(params, _, metadata['tLs'][0])
            best_tree = discretize_tree_topology(t_,n_all)
            best_seq = update_seq(params, seqs, metadata['seq_temp'])
            
            ## Log the metrics
        if(_%200==0 and args['log_wandb']):
            wandb.log(
                {"epoch":_, 
    
                "loss":loss[pos],
                "traversal cost": cost[pos],
                "traversal cost (surrogate)": cost_surrogate[pos],
                "tree force loss": tree_force_loss[pos],
                
                "tree":wandb.Image(tree_at_epoch),
                "tree matrix":wandb.data_types.Plotly(tree_matrix_at_epoch),
                "Seq":wandb.data_types.Plotly(new_seq),
                "last ancestor" :  wandb.data_types.Plotly(px.imshow((seqs_[-1]), text_auto=True)),
                "tLs":metadata['tLs'][0],})
    
        if(_%200==0):
            print_bold_info(f"epoch {_}")
            print("{:.3f}".format(metadata['tLs'][0]), end = " ")
            print("{:.3f}".format(loss[pos].item()), end=" ")
            print("{:.3f}".format(cost[pos].item()), end = "\n")
        
    
        ### update the tree loss schedule
        if(_%metadata['tLs'][3]==0):
            metadata['tLs'][0] = min(metadata['tLs'][2], metadata['tLs'][0] + metadata['tLs'][1])
            
    print_success_info("Optimization done!\n")
    print_success_info("Final cost: {:.5f}\n".format(cost[pos]))
    print_success_info("Best cost encountered: {:.5f}\n".format(best_ans))
    
    if(args['fix_tree']):
        print_success_info("Sankoff cost for groundtruth tree: {:.5f}\n".format(sankoff_cost.item()))
        target_cost = sankoff_cost
    elif(args['fix_seqs']):
        print_success_info("Groundtruth tree cost: {:.5f}\n".format(gt_cost))
        target_cost = gt_cost
    else:
        print_critical_info("No groundtruth to compare to\n")
        target_cost = 0
    
    
    if(abs(target_cost - best_ans) == 0):
        print_success_info("Optimization succeeded! Reached groundtruth 🚀\n")
        if (args['log_wandb']): wandb.log({"success":True, "Error" : 0})
    else:
        if (args['log_wandb']): wandb.log({"success":False, "Error" : (best_ans - sankoff_cost)})
    
    best_tree_img = show_graph_with_labels(best_tree, n_leaves, True)
    best_tree_adj = px.imshow(best_tree, text_auto=True)
    
    if(args['log_wandb']):
        wandb.log({
                "best cost":best_ans, 
                "best_tree_adj" : wandb.data_types.Plotly(best_tree_adj),
                "best_tree" : wandb.Image(best_tree_img),
                "best_seq" : wandb.data_types.Plotly(px.imshow(jnp.argmax(best_seq, axis = 2), text_auto=True))
                })
    
        wandb.finish()
    
    #saving all the useful data on files
    
    gt_tree_class=adjmat_to_treeclass(gt_tree_copy)
    best_tree_class=adjmat_to_treeclass(best_tree.astype(int))
    
    gt_tree_newick=gt_tree_class.format("newick")
    best_tree_newick=best_tree_class.format("newick")
    
    gt_tree_ete3 = ete3.Tree(gt_tree_newick)
    best_tree_ete3 = ete3.Tree(best_tree_newick)
    
    gt_tree_dendropy = dendropy.Tree.get(
            data=gt_tree_newick,
            schema="newick")
    
    count=0
    binary_counting=0
    for nd in gt_tree_dendropy.postorder_node_iter():
        count+=1
        if len(nd._child_nodes) == 1 or len(nd._child_nodes) >2 :
            gt_nb_non_binary+=1
            break
        else: binary_counting+=1
    
    if binary_counting==count: 
        gt_nb_binary+=1
        quant_numbers_dict["gt_b1"].append(dendropy.calculate.treemeasure.B1(gt_tree_dendropy))
        quant_numbers_dict["gt_colless"].append(dendropy.calculate.treemeasure.colless_tree_imbalance(gt_tree_dendropy))    
    
    best_tree_dendropy = dendropy.Tree.get(
            data=best_tree_newick,
            schema="newick")
    
    count=0
    binary_counting=0
    for nd in best_tree_dendropy.postorder_node_iter():
        count+=1
        if len(nd._child_nodes) == 1 or len(nd._child_nodes) >2 :
            bt_nb_non_binary+=1
            break
        else: binary_counting+=1
    
    if binary_counting==count: 
        bt_nb_binary+=1
        quant_numbers_dict["best_tree_b1"].append(dendropy.calculate.treemeasure.B1(best_tree_dendropy))
        quant_numbers_dict["best_tree_colless"].append(dendropy.calculate.treemeasure.colless_tree_imbalance(best_tree_dendropy))
    
    Phylo.draw(gt_tree_class, do_show=False)
    plt.title("GT Tree \n leaves=%i seqlength=%i mutations=%i letters=%i" %(n_leaves,seq_length,n_mutations,n_letters))
    plt.savefig(folder_name+'/images/sim%i_gt_tree.png'%n_simul)
    
    Phylo.draw(best_tree_class, do_show=False)
    plt.title("Best Tree \n leaves=%i seqlength=%i mutations=%i letters=%i" %(n_leaves,seq_length,n_mutations,n_letters))
    plt.savefig(folder_name+'/images/sim%i_best_tree.png'%n_simul)
    
    best_trees.append(best_tree.astype(int))
    gt_trees.append(gt_tree_copy)
    gt_sequences.append(gt_seqs)
    best_sequences.append(jnp.argmax(best_seq, axis = 2))
    
    quant_numbers_dict["gt_cost"].append(gt_cost_copy.tolist())
    quant_numbers_dict["sankoff_cost"].append(sankoff_cost.tolist())
    quant_numbers_dict["best_cost"].append(best_ans.tolist())
    quant_numbers_dict["gt_sackin"].append(dendropy.calculate.treemeasure.sackin_index(gt_tree_dendropy, normalize=True))
    quant_numbers_dict["best_tree_sackin"].append(dendropy.calculate.treemeasure.sackin_index(best_tree_dendropy, normalize=True))
    quant_numbers_dict["rf_gt_best_tree"].append(gt_tree_ete3.robinson_foulds(best_tree_ete3, unrooted_trees=True, )[0])

quant_numbers_dict["gt_nb_binary"]=gt_nb_binary
quant_numbers_dict["gt_nb_non_binary"]=gt_nb_non_binary
quant_numbers_dict["bt_nb_binary"]=bt_nb_binary
quant_numbers_dict["bt_nb_non_binary"]=bt_nb_non_binary
        
np.savez(folder_name+'/matrices.npz',
         best_trees=best_trees,
         gt_trees=gt_trees,
         gt_sequences=gt_sequences,
         best_sequences=best_sequences
         )  
        
with open(folder_name+'/quant_numbers.pickle', 'wb') as handle:
    pickle.dump(quant_numbers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
