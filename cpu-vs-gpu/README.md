# What is the performance difference when using a CPU vs a GPU vs 2 GPUs

## Setup

1. The scripts used the defaults as much as posible.
2. The scripts use as close to identical code as posible and are read in the normal fassion.
3. The version of Python was 3.8.2 64-bit with TensorFlow 2.2.
4. The scripts were run on an Intel Core i7-6700K with 64GB of Corsair Vengeance LPX DDR4 Ram and upto 2 NVIDIA GeForce GTX 1070 video cards.
5. Only the model training time was considered in results.
6. The scripts were run from VS Code without debugging (Ctrl+F5).
7. A new Python session was used between each run.
8. [Open Hardware Monitor](https://openhardwaremonitor.org/) was used to report the Max GPU and CPU

## Results

Below can be found the results for the different scripts.
Using two GPUs at the same time was evaluated but the results discarded because that configuration lacks key support on Windows.
The code works, but in order to combine the results, the CPU needs to get involved (evaluated as slow) or signifficently different code needs written (not evaluated).
At some point in the future I will re-evaluate the results and upgrade the code.

settings = batch_size/epochs/sample_length/units

| settings | no. GPU | Loss | Time (min) | Max CPU % | Max GPU 1 % | Max GPU 2 % |
|---|---|---|---|---|---|---|
|   1/10/11/40 | 0 | 20.067 | 02:25:29 | 78.7 | 0 |  7 |
|   1/10/11/40 | 1 | 21.601 | 01:29:50 | 48.0 | 0 | 26 |
|   4/10/11/40 | 0 |  5.463 | 00:42:11 | 78.9 | 0 | 40 |
|   4/10/11/40 | 1 |  5.378 | 00:27:05 | 41.8 | 0 | 92 |
|  16/10/11/40 | 0 |  3.013 | 00:17:05 | 62.9 | 0 | 18 |
|  16/10/11/40 | 1 |  3.645 | 00:10:16 | 62.9 | 0 | 88 |
|  64/10/11/40 | 0 |  4.629 | 00:09:18 | 68.8 | 0 | 70 |
|  64/10/11/40 | 1 |  4.641 | 00:06:36 | 28.4 | 0 | 41 |
| 128/10/11/40 | 0 |  4.879 | 00:07:43 | 58.6 | 0 | 29 |
| 128/10/11/40 | 1 |  4.959 | 00:05:56 | 33.2 | 0 | 69 |
| 256/10/11/40 | 0 |  5.471 | 00:07:06 | 55.8 | 0 | 16 |
| 256/10/11/40 | 1 |  5.483 | 00:05:34 | 39.3 | 0 | 25 |

## Conclusion

Using the GPU is ~20% faster on small simple networks.
