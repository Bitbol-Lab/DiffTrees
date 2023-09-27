In this folder the sequences have a fixed number (`m`) of aleatory mutations and the ancestral root is an aleatory sequence of length 198, where 1000 mutations have been introduced to create an already low energy root . At each bifurcation, the Hamming Distance between parent-child can be lower than `m`, (possible to mutate the same index twice but forced to mutate in another state), even without fitness.

In this folder, `sl` has to be 198, and we can either choose an alphabet size of 2, or 20. 

We introduce fitness through the use of Markov chain Monte Carlo method (MCMC)

## Field ##
* 2 letters: Field is sampled from a normal distribution N(µ=1, σ=1)
* 20 letters alphabet: Field is inferred from beta-lactamase2

## Coupling ##
The Coupling matrices are symmetric, this is why in the code we only iterate on the corresponding first dimension.
* 2 letters: Coupling is created with a binary matrix. Probability that 2 positions are connected ("1"), each in a certain state is 2%. Associated Temperature = 5
* 20 letters alphabet: Coupling is inferred from beta-lactamase2. Associated Temperature=1

### Example : running for trees with 16 leaves without fitness ###

```bash
python3 train_batch_implicit_diff.py -l 16 -nl 20 -m 50 -sl 198 -tLs [0,0.005,10,50] -lr 0.1 -lr_seq 0.01 -t float64-multi-init-run -p Batch-Run-Maximum-Parsimony -alt -n "Final Run" -g 0 -e 5000 -ai 1 -ic 50
```

### Example : running for trees with 16 leaves with field component ###

```bash
python3 train_batch_implicit_diff.py -l 16 -nl 20 -m 50 -sl 198 -tLs [0,0.005,10,50] -lr 0.1 -lr_seq 0.01 -field -t float64-multi-init-run -p Batch-Run-Maximum-Parsimony -alt -n "Final Run" -g 0 -e 5000 -ai 1 -ic 50
```

### Example : running for trees with 16 leaves with coupling component ###

```bash
python3 train_batch_implicit_diff.py -l 16 -nl 20 -m 50 -sl 198 -tLs [0,0.005,10,50] -lr 0.1 -lr_seq 0.01 -coupling-t float64-multi-init-run -p Batch-Run-Maximum-Parsimony -alt -n "Final Run" -g 0 -e 5000 -ai 1 -ic 50
```

### Example : running for trees with 16 leaves with both field and coupling ###

```bash
python3 train_batch_implicit_diff.py -l 16 -nl 20 -m 50 -sl 198 -tLs [0,0.005,10,50] -lr 0.1 -lr_seq 0.01 -field -coupling -t float64-multi-init-run -p Batch-Run-Maximum-Parsimony -alt -n "Final Run" -g 0 -e 5000 -ai 1 -ic 50
```
## Output ##
When running the code, it will create several trees (currently set to 100). 
It will create a file named respecting this example `16leaves_seql198_20letters_50muts_Field_True_Coupling_True` 
In this file you will find an 
* `Images` folder containing all the trees in a png format
* `matrices.npz` containing the relevant matrices (groundtruth sequences, best final sequences, groundtruth tree, best final tree)
* `quant_numbers.pickle` a dictionary containing relevant evaluation numbers on the 100 generated trees. (please note that some balance metrics, such as B1 and Colless are only performed on binary trees, therefore you might not find 100 numbers in the corresponding array. However, you can check how many trees in the simulations are binary and non binary with the keys `"bt_nb_binary"` and `"bt_nb_non_binary"`)
