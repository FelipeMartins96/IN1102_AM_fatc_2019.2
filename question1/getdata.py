import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

def getdata(normalize=True, remove_0=True):
    # Read data from Image Segmentation Database
    data = pd.read_csv('question1/data/seg.test')

    # Splits into shape view and rgb view
    # First 9 features
    # shape_view([2100]points (n), [9]features (p))
    shape_view = data.values[:, 0:9]
    # 10 Remaining features
    # rgb_view([2100]points (n), [10]features (p))
    rgb_view = data.values[:, 9:19]

    if remove_0:
        # Remove sigma = 0 features
        shape_view = shape_view[:,[0,1,3,5,6,7,8]]

    if normalize:
        # Normalize data
        rgb_view = minmax_scale(rgb_view, feature_range=(0, 1), axis=0)
        shape_view = minmax_scale(shape_view, feature_range=(0, 1), axis=0)

    return {'rgb': rgb_view, 'shape': shape_view}