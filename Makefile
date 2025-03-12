CC = gcc
CFLAGS = -lm -fopenmp -lopenblas -Ofast

neural_net:
	${CC} neural_net_cpu.c -o neural_net_cpu_native ${CFLAGS}
	${CC} neural_net_cpu.c -o neural_net_cpu_blas -DUSE_CBLAS ${CFLAGS}
	nvcc -arch=compute_70 neural_net_gpu.cu -o neural_net_gpu_native -Xcompiler -fopenmp -O3
	nvcc -arch=compute_70 neural_net_gpu.cu -o neural_net_gpu_cublas -Xcompiler -fopenmp -lcublas -DUSE_CUBLAS -O3

clean:
	rm -f neural_net_cpu_native neural_net_cpu_blas neural_net_gpu_native neural_net_gpu_cublas *.txt
