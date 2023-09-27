import math
import jax.numpy as jnp
from jax import vmap
from jax import random
from jax import jit
from functools import partial

import numpy as np
from typing import Dict, List
from jaxtyping import Array, Float
import copy

import math

from random import randint
from math import exp
import copy

target_depth = 0
num_mutations = 0
sequences = []
n_letters = 0


##new functions
def mcmc(_n_mutations, L_Spin, _seq_length, _n_letters, _Field, _Coupling) :  
    
    "function to flip one sequence (L_Spin)"
    
    c_mutation=0
    if _n_letters==2:Temp=float(5.0)
    if _n_letters==20:Temp=float(1.0)
    
    while c_mutation<_n_mutations:   
        selected_node = randint(0,_seq_length-1)
        new_state = randint(0,_n_letters-1)
        old_state=L_Spin[selected_node]
        while new_state==old_state:
            new_state=randint(0,_n_letters-1)
        old_L_Spin=L_Spin.copy()
        L_Spin[selected_node]= new_state
        de = (
            Pseudo_Hamiltonian(selected_node, new_state, L_Spin, _Field, _Coupling) -
            Pseudo_Hamiltonian(selected_node, old_state, old_L_Spin, _Field, _Coupling)
             )
        if de>=0 or np.random.rand()<exp(de/Temp):
            #already done above:
            #L_Spin[selected_node]= new_state
            c_mutation += 1
        else:
            L_Spin[selected_node]= old_state
    
            

def Pseudo_Hamiltonian(_node, _state_node, L_Spin, _Field, _Coupling) :
    
    """"
    Function to calculate the hamiltonian only using field and Jij
    Here, for the L_Spin component we multiply first Jij with L_Spin, which gives us
    a seq_length x 1 matrix, where each row is the sum of the 
    """
    
    #we substract the diagonal element that will be added in the for loop just after    
    hamiltonian = _Field[_node,_state_node]- _Coupling[_node,_node,_state_node,L_Spin[_node]]
        
    #we only iterate on one "line", because matrix is built symetrically, so one line
    #contains all infos already (repetitive to also check corresponding column)
    for i in range(L_Spin.shape[0]):
        hamiltonian += _Coupling[_node,i,_state_node,L_Spin[i]]
    
    return hamiltonian


def fitness_field_msa(seq_length,n_mutations,n_letters,n_leaves, 
                             flip_before_start, field_fitness, coupling_fitness):
    
    """
    function creating a strictly binary msa in the format of the paper
    index 0 to n_leaves in msa is the number of leaves, last index is root of tree
    because we create the binary symetric tree separately, rules are <strict>
    which makes it easier to create the msa
    """
    
    n_ancestors=n_leaves-1
    n_total=n_leaves*2-1
    n_generations=int(math.log2(n_leaves))
    #if initial sequence not at equilibrium, set flip_before_start=0

    #we chose the option (2 letters, 20 letters, with/without field/coupling)
    if n_letters==2:
        if field_fitness: 
            Field=np.load("Hs_and_Js/Field_2_states.npy")
        else:
            Field=np.zeros((seq_length,2))
        if coupling_fitness: 
            Coupling=np.load("Hs_and_Js/Coupling_2_states.npy")
        else:
            Coupling=np.zeros((seq_length,seq_length,2,2))
        
    elif n_letters==20:
        if field_fitness: 
            Field=np.load("Hs_and_Js/Field_20_states.npy")
        else:
            Field=np.zeros((seq_length,20))
        if coupling_fitness: 
            Coupling=np.load("Hs_and_Js/Coupling_20_states.npy")
        else:
            Coupling=np.zeros((seq_length,seq_length,20,20))
    
    else: print("Fitness only works with either 2 or 20 letters, sorry")
        
        
    l_spin = np.random.randint(0,high=n_letters,size = (seq_length),dtype = np.int8)

    msa = np.zeros((n_total,seq_length),dtype = np.int8)

    old_l_spin=copy.deepcopy(l_spin)

    mcmc(flip_before_start, l_spin, seq_length, n_letters, Field, Coupling)

    # print("difference between aleatory and equilibrium = ", np.count_nonzero(l_spin!=old_l_spin))

    msa[-1] = l_spin


    for i in range(1,n_leaves):
        # print(-2*i % len(msa), "have root", -i % len(msa))
        # print((-2*i-1) % len(msa), "have root", -i % len(msa))
        msa[-2*i] = msa [-i]
        msa[-2*i-1] = msa[-i]
        mcmc(n_mutations,msa[-2*i], seq_length, n_letters, Field, Coupling)
        mcmc(n_mutations,msa[-2*i-1], seq_length, n_letters, Field, Coupling)
        
    #     print("differences between root %i and leaf %i is = %i" %( -i % len(msa), -2*i % len(msa),np.count_nonzero(msa[-2*i] != msa [-i])))
    #     print("differences between root %i and leaf %i is = %i" %( -i % len(msa), (-2*i-1) % len(msa),np.count_nonzero(msa[-2*i-1] != msa [-i])))
            

    # for n in range(n_leaves):
    #      print("differences between root %i and leaf %i is = %i" %( -1 % len(msa), n,np.count_nonzero(msa[-1] != msa [n])))
        
    return msa


##

def mutate(key, exclude_indexes):
  '''
    Mutates a sequence by randomly changing one of the letters

    Accepts:
      exclude_indexes: list of indexes to exclude from the mutation (shape = (n_mutations,1))

    Returns:
      mutated sequence
  '''
  global n_letters

  space = jnp.tile(jnp.arange(n_letters), (num_mutations, 1)) 
  options  = space[space != exclude_indexes].reshape((space.shape[0],space.shape[1] - exclude_indexes.shape[1]))

  key_ = random.split(key, num_mutations)

  mutations = vmap(random.choice)(key_, options)

  return jnp.expand_dims(mutations, axis=1)

def create_seq(key, seq, depth):

  k1, k2 = random.split(key)
  
  if(depth > target_depth):
    return

  random_indexes = [[] for x in range(2)]
  
  random_indexes[0]= random.choice(k1, jnp.array(range(len(seq))), (num_mutations,1), replace=False)
  random_indexes[1]= random.choice(k2, jnp.array(range(len(seq))), (num_mutations,1), replace=False)


  seq_1 = seq
  seq_2 = seq

  seq_1 = seq_1.at[random_indexes[0]].set(mutate(k1, seq[random_indexes[0]]))
  seq_2 = seq_2.at[random_indexes[1]].set(mutate(k2, seq[random_indexes[1]]))


  #seq
  sequences[depth].append(seq_1)
  create_seq(k1, seq_1, depth+1)
  
  #new_seq
  sequences[depth].append(seq_2)
  create_seq(k2, seq_2, depth+1)


def generate_groundtruth(metadata : Dict[str, int], seed_, verbose = False) -> List[Array]: 
    '''
        Generates a groundtruth example based on the metadata provided


        Args : 
            metadata: dictionary containing the required specifications
            seed: random seed
            verbose: print the number of leaves generated

        Returns:
            masked_main : masked sequences (shape = (n_all, seq_length))
            true_main   : true sequences   (shape = (n_all, seq_length))
            tree        : tree structure   (shape = (n_all, n_all))
    '''

    key = random.PRNGKey(seed_)

    global target_depth
    global num_mutations
    global sequences
    global n_letters

    num_mutations = metadata['n_mutations']
    n_all       = metadata['n_all']
    n_leaves    = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']
    seq_length  = metadata['seq_length']
    n_letters   = metadata['n_letters']
    
    fitness_msa=fitness_field_msa(seq_length,
                                  num_mutations,
                                  n_letters,
                                  n_leaves, 
                                  flip_before_start=1000,
                                  field_fitness=metadata['fitness_field'],
                                  coupling_fitness=metadata['fitness_coupling']
                                  )
    new_seq_leaves=jnp.array(fitness_msa[0:n_leaves]).astype(jnp.bfloat16)
    true_main=jnp.array(fitness_msa).astype(jnp.bfloat16)
    masked_main=jnp.array(fitness_msa).astype(jnp.bfloat16)
    
    for i in range(n_leaves,len(true_main)): 
        masked_main=masked_main.at[i].set(jnp.zeros(seq_length,dtype=jnp.bfloat16))
    
    # target_depth = int(math.log2(n_leaves))-1

    # seq = jnp.zeros(seq_length, dtype=jnp.int64)

    # sequences = [[] for x in range(target_depth + 1)]
    # create_seq(key,seq,0)

    # if(verbose):
    #   print(len(sequences[target_depth]))

    # ### copy the leaves
    # masked_main = sequences[target_depth].copy()
    # true_main   = sequences[target_depth].copy()

    # n_leaves    = len(masked_main)
    # n_ancestors = n_leaves - 1


    # for i in range(0,n_ancestors):
    #     masked_main.append(seq)

    # ## Generate the true sequences by traversing through the leaves to the root
    # for i in range(len(sequences)-2, -1, -1):  #commence à -2 car déjà copié les leaves (sequences[-1])
    #     for k in range(0,len(sequences[i])):
    #         true_main.append(sequences[i][k])
    # true_main.append(seq)

    # if(verbose):
    #   i = 0
    #   for item in true_main:
    #     print(">seq"+str(i))
    #     for char in item:
    #       print(char, end="")
    #     print()

    #     i+=1

    tree = []

    iter = n_ancestors

    for i in range(0,len(masked_main)):
        leave = [0]*len(masked_main)

        if(i%2==0 and i!=len(masked_main)-1):
            iter +=1 

        leave[iter] = 1
        tree.append(leave)
           
    gt_tree_copy=copy.deepcopy(tree)
    # print(
    #     #"type of masked_main:", masked_main.type,'\n'
    #     "length of masked_main:", len(masked_main),'\n','\n'
    #     #"type of true_main:", true_main.type,'\n'
    #     "length of true_main:", len(true_main),'\n','\n'
    #     #"type of tree:", tree.type,'\n'
    #     "length of tree:", len(tree),'\n'
    #       )

    return jnp.array(masked_main).astype(jnp.bfloat16), jnp.array(true_main).astype(jnp.bfloat16), jnp.array(tree).astype(jnp.bfloat16), gt_tree_copy
