import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_rand_score

if len(sys.argv) is not 3:
    print("usage: python result_analyser [dir name or gt] [shape, rgb or view]")
    sys.exit()

# Read data from Image Segmentation Database
data = pd.read_csv('data/seg.test')
ground_truth = pd.read_csv('data/test_gt.csv', header = None)[0]
# If gt it will output the groundtruth data from the dataset
if sys.argv[1] == 'gt':
    crisp = ground_truth
else:
    crisp = pd.read_csv(sys.argv[1] + "/best_u.csv", header=None).idxmax()


# # view([2100]points (n), features (p))
if sys.argv[2] == 'view':
    view = data.values[:, 0:19]
elif sys.argv[2] == 'shape':
    view = data.values[:, 0:9]
elif sys.argv[2] == 'rgb':
    view = data.values[:, 9:19]
else:
    print("usage: python result_analyser [dir name or gt] [shape, rgb or view]")
    sys.exit()

# Normalize data
view = minmax_scale(view, feature_range=(0, 1), axis=0)

if sys.argv[1] != 'gt':
    J = pd.read_csv(sys.argv[1] + "/best_J.csv", header=None).values[:]
    print("Error (J): %f" % J)
rand = adjusted_rand_score(ground_truth.values, crisp.values)
print("Adjusted rand index: " + str(rand))
for i in range(7):
    print("Number of points in cluster " + str(i+1) + ": " + str(np.count_nonzero(crisp.values == i)))



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(view)


principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, crisp], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title(sys.argv[2] +' PCA Clusters', fontsize = 20)
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
fig.savefig(sys.argv[1] + '/' + sys.argv[2] + '.png')