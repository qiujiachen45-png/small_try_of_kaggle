
from hardship1 import db_train,db_test
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from hardship1 import data_cleaning


y = db_train['Credit Default']  # 确保 y 是 Series，一维
X=db_train


X=data_cleaning(X)
#此函数的意义是通过互相关检验因变量和自变量的相关性，筛选出前五个特征
def mutual_relationship(X1,y):

    mi_score11=mutual_info_regression(X1,y,discrete_features=False)
    mi_score11=pd.DataFrame(mi_score11,columns=["Credit Default"])
    mi_score11=mi_score11.sort_values(by="Credit Default",ascending=False)
    the_chosen1=mi_score11.head(5).index.tolist()
    top_features=X1.columns[the_chosen1].tolist()
    print(top_features)


mutual_relationship(X,y)




