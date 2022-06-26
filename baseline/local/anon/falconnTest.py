import numpy as np
import falconn
import _falconn as _internal
dataset = np.array([[1,2,3,4,5],[4,5,6,7,8]])   
dataset = dataset.astype(np.float32)
dataset -= np.mean(dataset, axis=0)
print("#########")
print(dataset)
print("#########")
num_points, dim = dataset.shape
parms = falconn.get_default_parameters(num_points, dim)
lsh_index = falconn.LSHIndex(parms)
lsh_index.setup(dataset)
print(lsh_index._interal) 
print("#########")
print(dataset)
print("#########")






