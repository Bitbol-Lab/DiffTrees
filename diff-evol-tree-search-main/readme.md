#### **Example : running for trees with 16 leaves**

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

