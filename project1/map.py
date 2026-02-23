import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. 生成示例数据（二维）
# -----------------------
np.random.seed(42)
# x1, x2 高度相关
x1 = np.random.normal(0, 2, 100)
x2 = x1 * 0.8 + np.random.normal(0, 0.5, 100)

X = np.column_stack((x1, x2))

# -----------------------
# 2. 数据中心化
# -----------------------
X_mean = X - X.mean(axis=0)

# -----------------------
# 3. 协方差矩阵
# -----------------------
C = X_mean.T @ X_mean / X_mean.shape[0]
print("协方差矩阵 C:\n", C)

# -----------------------
# 4. 特征分解 (eig)
# -----------------------
eig_vals, eig_vecs = np.linalg.eig(C)
print("特征值:", eig_vals)
print("特征向量:\n", eig_vecs)

# -----------------------
# 5. 绘制数据和主成分方向
# -----------------------
plt.figure(figsize=(6,6))
plt.scatter(X_mean[:,0], X_mean[:,1], alpha=0.5)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

# 画主成分向量
for i in range(len(eig_vals)):
    vec = eig_vecs[:,i] * np.sqrt(eig_vals[i]) * 3  # 缩放便于显示
    plt.arrow(0, 0, vec[0], vec[1], color='red', width=0.05, head_width=0.2)
    plt.text(vec[0]*1.1, vec[1]*1.1, f'PC{i+1}', color='red', fontsize=12)

plt.xlabel("X1 (中心化)")
plt.ylabel("X2 (中心化)")
plt.title("2D 数据 + PCA 主成分方向")
plt.axis('equal')
plt.grid(True)
plt.show()
