import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
df = pd.read_csv("../datasets/train.csv")
num = df.select_dtypes(include=np.number).columns.tolist()
matplotlib.use('TkAgg')
cols = 3
rows = (len(num) // cols) + 1

# 1. 调整画布：高度设为 rows * 5 即可，不要太大
plt.figure(figsize=(15, rows * 5))

for i, feature in enumerate(num, 1):
    if feature == "SalePrice":
        continue

    ax = plt.subplot(rows, cols, i)

    # 2. 【核心】强制子图比例：1.2 表示高度是宽度的 1.2 倍
    # 如果想更长，就改成 1.5；如果觉得太长了，改成 1.0 (正方形)
    ax.set_box_aspect(1.2)

    sns.scatterplot(x="SalePrice", y=feature, data=df,
                    hue="SalePrice", palette="Blues",
                    edgecolor='w', alpha=0.3, s=20)  # 调小点的大小 s=20

    # 3. 标签放在内部
    plt.text(0.05, 0.9, feature, transform=ax.transAxes,
             fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.ylabel(feature, size=8)
    plt.legend([], [], frameon=False)

# 4. 间距控制：h_pad 是关键，增加上下间距
plt.tight_layout(pad=3.0, h_pad=4.0)

# 5. 【强烈建议】保存一份到本地，查看不受 PyCharm 缩放影响的真实效果

plt.savefig("my_long_plot.png", dpi=400, bbox_inches='tight')
print("图片已保存到项目文件夹下，请直接双击打开查看原图！")
