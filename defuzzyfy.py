import pandas as pd
import numpy as np
import sys

if len(sys.argv) is not 2:
    print("usage: python defuzzyfy.py [dir name]")

df = pd.read_csv(sys.argv[1] + "/best_u.csv", header=None)

# crisp = df.idxmax().values
# np.savetxt(sys.argv[1] + "/crisp.csv", crisp, delimiter=",")

new_df = pd.DataFrame()
for feature in df.columns:
    max = df[feature].max()
    new_df[feature] = df[feature].apply(lambda x: 1 if x == max else 0)
new_df.to_csv(sys.argv[1] + '/crisp.csv', index=False, header=False)

