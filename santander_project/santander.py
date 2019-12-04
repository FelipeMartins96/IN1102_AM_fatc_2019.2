from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import sys


#train = '/home/george/Documents/Mestrado/IF699/cleber/projeto_santander/data/train.csv'
#test = '/home/george/Documents/Mestrado/IF699/cleber/projeto_santander/data/test.csv'

train = sys.argv[1]
test = sys.argv[2]

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

# Create new features

idx = features = train_df.columns.values[2:202]
for df in [test_df, train_df]:
    df['sum'] = df[idx].sum(axis=1)
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['median'] = df[idx].median(axis=1)

# Normalize features
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
train_x = reduce_mem_usage(train_df[features])
train_y = train_df['target']
test_x = reduce_mem_usage(test_df[features])

scaler = RobustScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

# Split train and test inside train

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Feature selection using tree

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf = clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)

# Models test

# rf = RandomForestClassifier(n_estimators=100, n_jobs=4).fit(X_train, y_train)
# rf_prediction = rf.predict(X_test)
# print('X_train')
# print(accuracy_score(y_test, rf_prediction))
#0.8976

# rf = RandomForestClassifier(n_estimators=100, n_jobs=4).fit(X_train_selected, y_train)
# rf_prediction = rf.predict(X_test_selected)
# print('X_train_selected')
# print(accuracy_score(y_test, rf_prediction))
#0.8980

# lr = LogisticRegression(
#     C=1, random_state=0, solver='lbfgs', max_iter=500, n_jobs=4, 
#     multi_class='multinomial').fit(X_train, y_train)
# lr_prediction = lr.predict(X_test)
# print('X_train')
# print(accuracy_score(y_test, lr_prediction))
#0.9144666666666666

# lr = LogisticRegression(
#     C=1, random_state=0, solver='lbfgs', max_iter=500, n_jobs=4, 
#     multi_class='multinomial').fit(X_train_selected, y_train)
# lr_prediction = lr.predict(X_test_selected)
# print('X_train_selected')
# print(accuracy_score(y_test, lr_prediction))
# 0.90715

# lgb = LGBMClassifier(random_state=17, num_leaves=63, max_depth=-1,
#                      learning_rate=0.1, n_estimators=200)
# lgb.fit(X_train, y_train)
# print('X_train')
# print(accuracy_score(y_test, lgb.predict(X_test)))
#0.9149666666666667

# lgb = LGBMClassifier(random_state=17, num_leaves=63, max_depth=-1,
#                      learning_rate=0.1, n_estimators=200)
# lgb.fit(X_train_selected, y_train)
# print('X_train_selected')
# print(accuracy_score(y_test, lgb.predict(X_test_selected)))
#0.9149666666666667

# # RandomizedSearch on Adaboost

ad = AdaBoostClassifier()
n_estimators_lst = [150, 200]
learning_rate_lst = [0.01, 0.05, 0.1, 0.5, 1]
param_dist = dict(n_estimators=n_estimators_lst, learning_rate =learning_rate_lst)

rand = RandomizedSearchCV(ad, param_dist, cv=3, scoring='roc_auc', n_iter=10)
rand.fit(X_train, y_train)
# print(rand.best_score_)
# print(rand.best_params_)

# # RandomizedSearch on LightBoost
# import lightgbm as lgb
# fit_params={"early_stopping_rounds":30, 
#             "eval_metric" : 'auc', 
#             "eval_set" : [(X_test_selected,y_test)],
#             'eval_names': ['valid'],
#             'verbose': 100,
#             'categorical_feature': 'auto'}

