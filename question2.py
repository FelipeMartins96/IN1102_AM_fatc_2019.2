import sys, os, warnings
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import wilcoxon

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h

# Read data from Image Segmentation Database
data = pd.read_csv('data/seg.test')

# Load ground thruth labels
ground_truth = np.genfromtxt('data/test_gt.csv', delimiter=',')

# Splits into shape view and rgb view
# First 9 features
# shape_view([2100]points (n), [9]features (p))
shape_view = data.values[:, 0:9]
# 10 Remaining features
# rgb_view([2100]points (n), [10]features (p))
rgb_view = data.values[:, 9:19]

L = 2
K = 7

prob_priori = np.zeros(K)
for i in range(0,K):
    prob_priori[i] = np.count_nonzero(ground_truth==i) / ground_truth.shape[0]

#GRID SEARCH CROSS VALIDATION - ENCONTRAR MELHOR NUMERO DE VIZINHOS
kb_1 = KNeighborsClassifier()
kb_2 = KNeighborsClassifier()


k_range = list(range(1,31))
param_grid = dict(n_neighbors=k_range)

grid_1 = GridSearchCV(kb_1, param_grid, cv=10, scoring='accuracy', n_jobs=6)
grid_2 = GridSearchCV(kb_2, param_grid, cv=10, scoring='accuracy', n_jobs=6)

grid_1.fit(shape_view, ground_truth)
grid_2.fit(rgb_view, ground_truth)


print(grid_1.best_params_['n_neighbors'])
print(grid_2.best_params_['n_neighbors'])

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30)
score_gb = []
score_kb = []

for train_index, test_index in rskf.split(shape_view, ground_truth):

    shape_train, shape_test = shape_view[train_index], shape_view[test_index]
    rgb_train, rgb_test = rgb_view[train_index], rgb_view[test_index]
    true_labels_train, true_labels_test = ground_truth[train_index], ground_truth[test_index]

    # print('view: ', view+1)
    # matrix = D_matrices[view]
    # print(rskf.get_n_splits(true_labels))

        # print('TRAIN: ', len(train_index))
        # print('TEST: ', len(test_index))

    #Classificador Bayesiano Gaussiano
    gb_1 = GaussianNB()
    gb_2 = GaussianNB()

    gb_1.fit(shape_train, true_labels_train)
    gb_2.fit(rgb_train, true_labels_train)

    pred_1 = gb_1.predict_proba(shape_test)
    pred_2 = gb_2.predict_proba(rgb_test)

    #Ensemble pela regra da soma
    gb_ensemble = np.argmax(((1-L)*(prob_priori) + pred_1 + pred_2), axis=1)

    #Score do Ensemble
    score_gb.append(np.equal(gb_ensemble, true_labels_test).sum() / true_labels_test.shape[0])

    #Classificador KNN Bayesiano
    kb_1 = KNeighborsClassifier(n_neighbors=grid_1.best_params_['n_neighbors'])
    kb_2 = KNeighborsClassifier(n_neighbors=grid_2.best_params_['n_neighbors'])

    kb_1.fit(shape_train, true_labels_train)
    kb_2.fit(rgb_train, true_labels_train)

    pred_1 = kb_1.predict_proba(shape_test)
    pred_2 = kb_2.predict_proba(rgb_test)

    #Ensemble pela regra da soma
    kb_ensemble = np.argmax(((1-L)*(prob_priori) + pred_1 + pred_2), axis=1)
    #score dos ensembles
    score_kb.append(np.equal(kb_ensemble, true_labels_test).sum() / true_labels_test.shape[0])

# Teste Wilcoxon nos 2 classificadores acima
w, p = wilcoxon(score_gb, score_kb)
print(w,p)

#Estatistica
stats_results = np.array([
    mean_confidence_interval(score_gb),
    mean_confidence_interval(score_kb),
    w,
    p])
print(stats_results)
    
hist_skb  = sns.distplot(score_kb, hist=True, kde=True,
    bins=int(50), color = 'green',
    hist_kws={'edgecolor':'black'},
    kde_kws={'linewidth': 3})
fig1 = hist_skb.get_figure()

hist_sgb  = sns.distplot(score_gb, hist=True, kde=True,
    bins=int(50), hist_kws={'edgecolor':'red'},
    kde_kws={'linewidth': 3})
hist_sgb.figure.savefig('hist_sgb.png')
