import sklearn
import numpy as np

def firstnList(n):
    num = 0.1
    while num < n:
        yield [num]
        num += 1.
     

def firstn(n):
    num = 0.
    while num < n:
        yield num
        num += 1.
      


# X = firstnList(10)
# y = firstn(10)

# print list(X)
# print list(y)

# from sklearn.neighbors import NearestNeighbors

# neigh = NearestNeighbors(n_neighbors=1)
# neigh.fit(X)

# print neigh.kneighbors(2.5)
# print neigh.kneighbors([2.4])
# print neigh.kneighbors([2.6])
# print neigh.kneighbors([3])


#samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
#samples = [[0.], [1.], [2.]]
samples = list(firstnList(10))
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(samples) 
#print(neigh.kneighbors([1., 1., 1.])) 
print(neigh.kneighbors(X=[1.5], return_distance=False)) 
print(neigh.kneighbors(X=[1.4], return_distance=False)) 
print(neigh.kneighbors(X=[1.6], return_distance=False)) 
print(neigh.kneighbors(X=[1.7], return_distance=False)) 
print(neigh.kneighbors(X=[2], return_distance=False, n_neighbors=3))
print(neigh.kneighbors(X=[2], n_neighbors=3))