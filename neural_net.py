import numpy as np
import matplotlib.pyplot as plt

epoch = np.arange(50) + 1
loss = np.loadtxt("output.txt")

plt.figure(figsize = (5,5))
plt.plot(epoch, loss, color = "blueviolet", linewidth = 1)
plt.xticks(np.arange(0, 55, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy Loss')
plt.title('Loss Curve of Benchmark Problem', size = 15, y = 1.025)
plt.grid(True, linestyle = "--", alpha = 0.5)
plt.show()
