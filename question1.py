from question1.getdata import getdata
from question1.getsigma import getsigma
from question1.initialize import initialize
from question1.update import update
from question1.getj import getj

# Number of Clusters
c = 7
# Fuzziness of membership
m = 1.6
# Iteration limit
T = 1
# Error threshold
e = 10e-10
# Number of Epochs
ep = 100

data = getdata()

for name, view in data.items():
    # Number of points
    n = view.shape[0]
    # Number of features
    p = view.shape[1]

    sigma = getsigma(view, p)

    for epoch in range(ep):
        print("epoch ", epoch+1)

        calc_data = initialize(c, n, p)

        J = float("inf")
        best_J = float("inf")

        for it in range(T):
            print("iteration ", it+1)

            calc_data = update(c, n, p, m, sigma, view, **calc_data)

            J_prev = J
            J = getj(c, n, p, m, sigma, view, **calc_data)
            print(J)
        
