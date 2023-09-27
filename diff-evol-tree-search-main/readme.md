In this folder the sequences have a fixed number (`m`) of aleatory mutations and the ancestral root is a sequence of length `sl` filled with zeros. At each bifurcation, the Hamming Distance between parent-child is exactly `m`, (`m` different indices mutate and forced to mutate in another state).

### Example : running for trees with 16 leaves with ###

* To run for different number of leaves change the -l to the desired value

Other params :

* sequence length : `-sl`
* mutations per bifurcation : `-m`
* alphabet size : `-nl`
* epochs/steps : `-e`
* initialization count to run in parallel : `-ic`

During running, every 200 steps it will print the `soft_parsimony_score` and `parsimony_score` (last two values in each line)

```bash
python3 train_batch_implicit_diff.py -l 16 -nl 20 -m 50 -sl 256 -tLs [0,0.005,10,50] -lr 0.1 -lr_seq 0.01 -t float64-multi-init-run -p Batch-Run-Maximum-Parsimony -alt -n "Final Run" -g 0 -e 5000 -ai 1 -ic 50
```

## Output ##
When running the code, it will create several trees (currently set to 100). 
It will create a file named respecting this example `16leaves_seql198_20letters_50muts_Field_True_Coupling_True` 
In this file you will find an 
* `Images` folder containing all the trees in a png format
* `matrices.npz` containing the relevant matrices (groundtruth sequences, best final sequences, groundtruth tree, best final tree)
* `quant_numbers.pickle` a dictionary containing relevant evaluation numbers on the 100 generated trees. (please note that some balance metrics, such as B1 and Colless are only performed on binary trees, therefore you might not find 100 numbers in the corresponding array. However, you can check how many trees in the simulations are binary and non binary with the keys `"bt_nb_binary"` and `"bt_nb_non_binary"`)
