# What is the performance difference in R vs Python?

## Setup

1. Both scripts used the defaults as much as posible.
2. Both scripts use as close to identical code as posible and are read in the normal fassion.
3. The R script was devoloped in R Studio and the Python script in VS Code.
4. The version of R was 3.6.3 64-bit while the version of Python was 3.7.7 64-bit.
5. Both scripts were run on an Intel Core i7-6700K with 64GB of Corsair Vengeance LPX DDR4 Ram and 2 NVIDIA GeForce GTX 1070 video cards (see note).
6. Only the model training time was considered in results.
7. Both scripts were run from the command line.
   ```{shell}
   d:
   cd d:\repos\MindMimicLabs\tensorflow-performance\r-vs-python
   rscript simple_rnn.r
   -- or --
   python simple_rnn.py
   ```
8. A fresh shell was opened betewwn each run.
9. [Open Hardware Monitor](https://openhardwaremonitor.org/) was used to report the Max GPU and CPU

## Results

Below can be found the results for the 2 different scripts.

**NOTE:** Both scripts were run **WITH CPU ONLY** because R [requires Conda](https://tensorflow.rstudio.com/installation/gpu/local_gpu/) to enable GPU support despite the logging output saying "Adding visible gpu devices: 0, 1"

### R

settings = batch_size/epochs/sample_length/units

| settings | Loss | Time (min) | Max CPU % | Max GPU 1 % | Max GPU 2 % |
|---|---|---|---|---|
| 1024/10/11/40 | 5.770 | 47.13 | 34.7 | 0 | 10 |
| 0256/10/11/20 | 5.343 | 37.17 | 31.1 | 6 | 21 |
| 0128/10/11/20 | 4.750 | 36.52 | 86.7 | 0 | 11 |
| 0064/10/11/20 | 4.422 | 38.41 | 35.5 | 6 | 07 |
| 0064/10/11/10 | 4.886 | 38.25 | 33.1 | 0 | 12 |

### Python

settings = batch_size/epochs/sample_length/units

| settings | Loss | Time (min) | Max CPU % | Max GPU 1 % | Max GPU 2 % |
|---|---|---|---|---|
| 1024/10/11/40 | 6.203 | 00:05:44 | 78.5 | 0 | 05 |
| 0256/10/11/20 | 5.950 | 00:05:54 | 79.3 | 0 | 15 |
| 0128/10/11/20 | 5.397 | 00:06:19 | 85.7 | 0 | 03 |
| 0064/10/11/20 | 4.367 | 00:07:53 | 82.0 | 0 | 02 |
| 0064/10/11/10 | 4.800 | 00:07:13 | 83.2 | 0 | 17 |

## Conclusion

There is some amount of loss in performance when using R.
Waiting on another Lab mate to confirm results.
