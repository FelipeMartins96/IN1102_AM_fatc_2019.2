import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from scipy.stats import wilcoxon


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

score_gb = []


rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30)
for train_index, test_index in rskf.split(shape_view, ground_truth):

    shape_train, shape_test = shape_view[train_index], shape_view[test_index]
    rgb_train, rgb_test = rgb_view[train_index], rgb_view[test_index]
    true_labels_train, true_labels_test = ground_truth[train_index], ground_truth[test_index]

    #Classificador Bayesiano Gaussiano
    gb_shape = GaussianNB()
    gb_rgb = GaussianNB()

    gb_shape.fit(shape_train, true_labels_train)
    gb_rgb.fit(rgb_train, true_labels_train)

    pred_shape = gb_shape.predict_proba(shape_test)
    pred_rgb = gb_rgb.predict_proba(rgb_test)


    #Ensemble pela regra da soma
    gb_ensemble = np.argmax(((1-L)*(prob_priori) + pred_shape + pred_rgb), axis=1)

    #Score do Ensemble
    score_gb.append(np.equal(gb_ensemble, true_labels_test).sum() / true_labels_test.shape[0])

w = wilcoxon(score_gb)
print(w)