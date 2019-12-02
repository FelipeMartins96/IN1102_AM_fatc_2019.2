import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# Read data from Image Segmentation Database
data = pd.read_csv('data/seg.test')

# Load ground thruth labels
ground_truth = pd.read_csv('data/test_gt.csv', header = None)

# view([2100]points (n), [19]features (p))
view = data.values[:, 0:19]


# Normalize data
view = minmax_scale(view, feature_range=(0, 1), axis=0)
print(view)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(view)


principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, ground_truth], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Shape View PCA Clusters', fontsize = 20)
targets = [0, 1, 2, 3, 4, 5, 6]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
fig.savefig('view.png')