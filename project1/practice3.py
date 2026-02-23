import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.core.pylabtools import figsize
from seaborn import kdeplot

df = pd.read_csv("../datasets/train.csv")
num= df.select_dtypes(include=np.number).columns.tolist()

zip1=np.log1p(df["SalePrice"])
plt.subplots(figsize=(8,6))
sns.displot(zip1,kde=True)
plt.show()