import math

# Calculate Kernel
def gaussian(x,v,sigma):
    return math.exp((-((x-v)**2))/(sigma))
