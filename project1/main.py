import matplotlib.pyplot as plt
import pandas as pd
from numpy.f2py.cfuncs import includes
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


db=pd.read_csv("../datasets/train.csv", encoding="utf-8")




num=db.select_dtypes(include=[np.number]).columns.tolist()
word=db.select_dtypes(include=["object","category"]).columns.tolist()
missing_data=db.isnull().sum()
print(missing_data.head(10))


num_transform=Pipeline([(
    "imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

word_transform=Pipeline([(
    "imputer",SimpleImputer(strategy="constant",fill_value="missing")
),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

preprocessor=ColumnTransformer(
    transformers=[("num",num_transform,num),("word",word_transform,word)],
    remainder="drop"
)


detailed_num=num_transform.fit_transform(db[num])
print(detailed_num)

detailed_word=word_transform.fit_transform(db[word])
print(detailed_word)