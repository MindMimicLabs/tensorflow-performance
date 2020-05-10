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

settings = batch_size/epochs/sample_length/units

| settings | no. GPU | Loss | Time (min) | Max CPU % | Max GPU 1 % | Max GPU 2 % |
|---|---|---|---|---|---|
| 16/10/11/40 | 0 | 3.470 | 00:15:37 | 86.7 | 0 | 45 |
| 16/10/11/40 | 1 | 3.304 | 00:07:40 | 40.1 | 0 | 43 |
| 16/10/11/40 | 2 | xxx | xxx | xxx | xxx | xxx |

## Conclusion


