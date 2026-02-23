import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果不需要可以删除）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建x轴数据
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# 计算y值
y_func = 1 - np.cos(x)  # f(x) = 1 - cos(x)
y_upper = x**2 / 2      # 上界: x^2/2
y_lower = np.zeros_like(x)  # 下界: 0

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(x, y_upper, 'b-', label='$y = x^2/2$ (上界)', linewidth=2, alpha=0.8)
plt.plot(x, y_lower, 'g-', label='$y = 0$ (下界)', linewidth=2, alpha=0.8)
plt.plot(x, y_func, 'r-', label='$y = 1 - \cos(x)$', linewidth=2)

# 填充夹逼区域
plt.fill_between(x, y_lower, y_upper, alpha=0.15, color='gray', label='夹逼区域')

# 设置坐标轴
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 设置标题和标签
plt.title('不等式图示：$0 \\leq 1 - \\cos(x) \\leq \\frac{x^2}{2}$')
plt.xlabel('x')
plt.ylabel('y')

# 设置坐标轴范围
plt.xlim([-2*np.pi, 2*np.pi])
plt.ylim([-0.5, 10])

# 添加图例
plt.legend()

# 添加网格
plt.grid(True, alpha=0.3)

# 显示图形
plt.tight_layout()
plt.show()