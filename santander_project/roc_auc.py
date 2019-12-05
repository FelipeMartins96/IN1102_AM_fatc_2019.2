from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os

figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

train = '/home/george/Documents/Mestrado/IF699/cleber/projeto_santander/data/train.csv'
test = '/home/george/Documents/Mestrado/IF699/cleber/projeto_santander/data/test.csv'

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

# Adaboost
ad = AdaBoostClassifier(n_estimators = 240, learning_rate= 0.5)
ad.fit(X_train, y_train)

# RandomForest
rf = RandomForestClassifier(n_estimators = 25, min_samples_split=10,
                            min_samples_leaf=8, max_features='auto',
                            max_depth=8, bootstrap=True)
rf.fit(X_train, y_train)

# LogisticRegression
lr = LogisticRegression(C = 10, penalty = 'l2').fit(X_train, y_train)

# LightGBM
lgb = LGBMClassifier(random_state=17, num_leaves=20, max_depth=2,
                     learning_rate=0.5, n_estimators=300, min_data_in_leaf=30)
lgb.fit(X_train, y_train)

# AUC

ad_probs = ad.predict_proba(X_test)
# keep probabilities for the positive outcome only
ad_probs = ad_probs[:, 1]
ad_auc = roc_auc_score(y_test, ad_probs)
# summarize scores
print('Adaboost: ROC AUC=%.3f' % (ad_auc))

rf_probs = rf.predict_proba(X_test)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
# summarize scores
print('RandomForest: ROC AUC=%.3f' % (rf_auc))

lr_probs = lr.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('LogisticRegression: ROC AUC=%.3f' % (ad_auc))

lgb_probs = lgb.predict_proba(X_test)
# keep probabilities for the positive outcome only
lgb_probs = lgb_probs[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_probs)
# summarize scores
print('LightGBM: ROC AUC=%.3f' % (ad_auc))


# calculate roc curves
ad_fpr, ad_tpr, _ = roc_curve(y_test, ad_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
lgb_fpr, lgb_tpr, _ = roc_curve(y_test, lgb_probs)
# plot the roc curve for the model
pyplot.plot(ad_fpr, ad_tpr, marker='.', label='Adaboost')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='RandomForest')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='LogisticRegression')
pyplot.plot(lgb_fpr, lgb_tpr, marker='.', label='LightGBM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# calculate precision-recall curve
# calculate roc curves
ad_precision, ad_recall, _ = precision_recall_curve(y_test, ad_probs)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lgb_precision, lgb_recall, _ = precision_recall_curve(y_test, lgb_probs)
# predict class values
ad_yhat = ad.predict(X_test)
rf_yhat = rf.predict(X_test)
lr_yhat = lr.predict(X_test)
lgb_yhat = lgb.predict(X_test)
ad_f1, ad_auc = f1_score(y_test, ad_yhat), auc(ad_recall, ad_precision)
rf_f1, rf_auc = f1_score(y_test, rf_yhat), auc(rf_recall, rf_precision)
lr_f1, lr_auc = f1_score(y_test, lr_yhat), auc(lr_recall, lr_precision)
lgb_f1, lgb_auc = f1_score(y_test, lgb_yhat), auc(lgb_recall, lgb_precision)
# summarize scores
print('Adaboost: f1=%.3f auc=%.3f' % (ad_f1, ad_auc))
print('RandomForest: f1=%.3f auc=%.3f' % (rf_f1, rf_auc))
print('LogisticRegression: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
print('LightGBM: f1=%.3f auc=%.3f' % (lgb_f1, lgb_auc))
# plot the precision-recall curves
pyplot.plot(ad_recall, ad_precision, linestyle='--', label='Adaboost')
pyplot.plot(rf_recall, rf_precision, linestyle='--', label='RandomForest')
pyplot.plot(lr_recall, lr_precision, linestyle='--', label='LogisticRegression')
pyplot.plot(lgb_recall, lgb_precision, linestyle='--', label='LightGBM')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()