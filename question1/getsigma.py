from scipy.spatial.distance import pdist
import numpy as np

def getsigma(view, p):
    sigma = []
    for j in range(p):
        dist = pdist(view[:,j].reshape(-1,1)) # Size given by Binominal Coefficient
        mean = np.mean([np.quantile(dist, 0.1), np.quantile(dist, 0.9)])
        sigma.append(mean)
    
    return sigma