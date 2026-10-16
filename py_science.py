import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y, label='sin', color='blue', linestyle='-')
plt.plot(x, y2, label='cos', color='red', linestyle='--')

plt.title("Trigonometric Functions")
plt.xlabel("X axis (Time/Steps)")   #加 X 轴标签
plt.ylabel("Y axis (Value)")        #加 Y 轴标签

plt.grid(True, linestyle=':', alpha = 0.6)
plt.legend(loc = 'upper right')

plt.savefig("sine_wave_paper.png", dpi = 300)
print("图像已保存")

