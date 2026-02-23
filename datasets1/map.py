import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("train.csv")
plt.subplots(figsize=(8,4))
sns.histplot(df['Current Loan Amount'])
plt.show()

