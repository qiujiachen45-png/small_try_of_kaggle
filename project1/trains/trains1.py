#探索性数据分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

#读取数据
df1=pd.read_csv("datasets/train.csv")

#SalePrice的分布图

safe=df1["SalePrice"]
plt.subplots(figsize=(8,4))
sns.displot(data=safe,height=4,aspect=8,palette="blues")
plt.show()

#一一对应

fea_number=df1.select_dtypes(include=[np.number]).columns.tolist()

col=len(fea_number)//3+1

plt.subplots(figsize=(col,3))
for i, fea1 in enumerate(fea_number,1):
    if fea1=="SalePrice":
        break
    plt.subplot()
    sns.displot(data=df1[fea1])
    plt.text(0.3,0.3,f"{fea1}")
plt.savefig("picture1.png")

#改造log

df1["SalePrice"]=np.log1p(df1["SalePrice"])


#找到多少不

def selection_missing(df):
    dict_i={}
    d1=list(pd.DataFrame(df))
    for i in range(len(d1)):
        dict_i.update({d1[i]:round(df1[d1[i]].isnull().mean()*100,2)})
    return dict_i


missing=selection_missing(df1)

df_missing=sorted(missing.items(),key=lambda x: x[1],reverse=True)
print(df_missing[0:10])


#数据清洗
df1["sale"]=






