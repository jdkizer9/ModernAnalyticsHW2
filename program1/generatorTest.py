import sklearn
import numpy as np
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1
        
print list(firstn(10))

a = np.array(firstn(10))