import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("train.csv")
sss=df.select_dtypes(include=["object"]).columns.tolist()
print(f"{sss}")