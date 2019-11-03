from sklearn.metrics import adjusted_rand_score
import numpy as np

ground_truth = np.genfromtxt('data/test_gt.csv', delimiter=',')
rgb_crisp = np.genfromtxt('rgb/crisp.csv', delimiter=',')
shape_crisp = np.genfromtxt('shape/crisp.csv', delimiter=',')

print('rand gt and rgb:    ' + str(adjusted_rand_score(ground_truth, rgb_crisp)))
print('rand gt and shape:  ' + str(adjusted_rand_score(ground_truth, shape_crisp)))
print('rand shape and rgb: ' + str(adjusted_rand_score(shape_crisp, rgb_crisp)))