The code is based on https://github.com/ramithuh/diff-evol-tree-search.git from Differentiable Search of Evolutionary Trees (Hettiarachchi, Swartz, Ovchinnikov, 2023). 

### requirements.txt ###
These are the requirements used when launching either one of the two files on SCITAS. 
I suggest installing jax libraries (especially with cuda, cudnn dependecies) using conda instead of pip.

### diff-evol-tree-search-main ###
This folder is a copy of the one present in the paper with some additional functionalities, such as result vizualisation without the wandb.ai intermediary

### diff-evol-tree-fitness ###
This folder introduces fitness (both a site specific field component and/or a pairwise interaction coupling)