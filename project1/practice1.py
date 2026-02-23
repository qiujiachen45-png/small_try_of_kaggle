import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


df = pd.read_csv("../datasets/train.csv")
num= df.select_dtypes(include=np.number).columns.tolist()

num2=df[num].corr()
plt.subplots(figsize=(15,12))
sns.heatmap(num2,vmax=0.9,cmap="Blues",square=True)
plt.show()