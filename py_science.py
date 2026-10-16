import numpy as np
import matplotlib.pyplot as plt

# 生成 x 轴的数据（从 -6 到 6）
x = np.linspace(-6, 6, 200)

# 1. 计算 Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# 2. 计算 ReLU
relu = np.maximum(0, x)

# 开始画图
plt.figure(figsize=(10, 5))

# 画 Sigmoid (蓝色)
plt.plot(x, sigmoid, label='Sigmoid', color='blue', linewidth=2)

# 画 ReLU (红色)
plt.plot(x, relu, label='ReLU', color='red', linewidth=2)

# 加坐标系辅助线
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

# 加标签、图例、网格
plt.title('Sigmoid vs ReLU Activation Functions')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 保存图片（如果是本地电脑，也可以把下面这行换成 plt.show()）
plt.savefig('activation_functions.png', dpi=300)
print("图像已保存为 activation_functions.png")

