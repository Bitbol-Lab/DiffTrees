In this folder the sequences have a fixed number (`m`) of aleatory mutations and the ancestral root is a sequence of length `sl` filled with zeros. At each bifurcation, the Hamming Distance between parent-child can be lower than `m`, (possible to mutate the same index twice but forced to mutate in another state), even without fitness.

In this folder, `sl` has to be 198, and we can either choose an alphabet size of 2, or 20. 

## Field ##
* 2 letters: Field is sampled from a normal distribution N(µ=1, σ=1)
* 20 letters alphabet: Field is inferred from beta-lactamase2

## Coupling ##
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
