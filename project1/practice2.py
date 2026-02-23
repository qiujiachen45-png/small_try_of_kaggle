import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


df = pd.read_csv("../datasets/train.csv")
num= df.select_dtypes(include=np.number).columns.tolist()
data5=pd.concat([df["OverallQual"],df["SalePrice"]],axis=1)
fig,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=df["OverallQual"],y="SalePrice",data=data5)
fig.axis(ymin=0,ymax=1000000)
plt.show()