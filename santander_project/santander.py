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


train = 'data/train.csv'
test = 'data/test.csv'

# train = sys.argv[1]
# test = sys.argv[2]

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

# ##### Random forest test for feature selection
# rf = RandomForestClassifier().fit(X_train, y_train)
# rf_prediction = rf.predict(X_test)
# print('RF without feature selection')
# print(accuracy_score(y_test, rf_prediction))
# print('')
# #0.8980666666666667

# rf = RandomForestClassifier().fit(X_train_selected, y_train)
# rf_prediction = rf.predict(X_test_selected)
# print('RF with feature selection')
# print(accuracy_score(y_test, rf_prediction))
# print('')
# #0.8980166666666667

# ##### Adaboost test for feature selection
# ad = AdaBoostClassifier().fit(X_train, y_train)
# ad_prediction = ad.predict(X_test)
# print('AD without feature selection')
# print(accuracy_score(y_test, ad_prediction))
# print('')
# #0.90525

# ad = AdaBoostClassifier().fit(X_train_selected, y_train)
# ad_prediction = ad.predict(X_test_selected)
# print('AD with feature selection')
# print(accuracy_score(y_test, ad_prediction))
# print('')
# #0.9053333333333333

# ##### LightGBM test for feature selection
# lgb = LGBMClassifier().fit(X_train, y_train)
# print('LGBM without feature selection')
# print(accuracy_score(y_test, lgb.predict(X_test)))
# print('')
# #0.9056333333333333

# lgb = LGBMClassifier().fit(X_train_selected, y_train)
# print('LGBM with feature selection')
# print(accuracy_score(y_test, lgb.predict(X_test_selected)))
# #0.90645

# ##### LogistRegression test for feature selection
# lr = LogisticRegression().fit(X_train, y_train)
# lr_prediction = lr.predict(X_test)
# print('LR without feature selection')
# print(accuracy_score(y_test, lr_prediction))
# print('')
# #0.9144833333333333

# lr = LogisticRegression().fit(X_train_selected, y_train)
# lr_prediction = lr.predict(X_test_selected)
# print('LR with feature selection')
# print(accuracy_score(y_test, lr_prediction))
# print('')
# #0.90715

# Models hyperparameter tuning

# ### Random Forest

random_grid = {'bootstrap': [True, False],
 'max_depth': [2, 5, 10, 20],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [100, 200, 300]}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf, param_distributions = random_grid, n_iter = 4,
    cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
print(rf_random.best_score_)
print(rf_random.best_params_)

### Adaboost

ad = AdaBoostClassifier()
n_estimators_lst = [150, 200, 220]
learning_rate_lst = [0.3, 0.5, 1]#, 1.2]
param_dist = dict(n_estimators=n_estimators_lst, learning_rate =learning_rate_lst)
# [Parallel(n_jobs=-1)]: Done  12 out of  12 | elapsed:  2.5min finished
# 0.9003285714285715



ad_random = RandomizedSearchCV(ad, param_dist, cv=3, scoring='roc_auc', n_iter=4)
ad_random.fit(X_train, y_train)
print(ad_random.best_score_)
print(ad_random.best_params_)
# 0.8738800694920262
# {'n_estimators': 220, 'learning_rate': 0.5}
