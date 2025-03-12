# Machine Learning for Image Classification

Name: Annabelle Huang

CNET ID: ahuang02

To build and run the neural network programs, use the command `make neural_net`.

### CPU Native

- Let `partition = caslake` and `cpus-per-task = 16`. Use the command `srun ./neural_net_cpu_native` in the batchfile.

### CPU BLAS

- Let `partition = caslake` and `cpus-per-task = 16`. Use the command `srun ./neural_net_cpu_blas` in the batchfile.

### GPU Native

- Let `partition = gpu`, `constraint = v100`, `gres = gpu:1`, and `cpus-per-task = 16`. Use the command `srun ./neural_net_gpu_native` in the batchfile for the V100 processor.

### GPU cuBLAS

- Let `partition = gpu`, `constraint = v100`, `gres = gpu:1`, and `cpus-per-task = 16`. Use the command `srun ./neural_net_gpu_cublas` in the batchfile for the V100 processor.
