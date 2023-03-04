import numpy as np

plain_array = [0,1,2,3,4,5,6,7]
np_array = np.array(plain_array)
print(np_array[0:2])
print(np_array[-1])

plain_array = [
    [0,1,2],
    [0,10,20],
    [0,100,200]
]

np_array = np.array(plain_array)
print(np_array[0:2])
print(np_array[:,1])
print(np_array[0:2,1])
print(np_array[-1])
print(np_array[:,1:2])
print(np_array[:,1])