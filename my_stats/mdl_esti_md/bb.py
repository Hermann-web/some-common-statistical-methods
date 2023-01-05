import numpy as np
g = np.array([0.1, 0.2, -2, 5, 0])

print((g<0).sum() + (g>1).sum())
print(( (g<0) + (g>1) ).sum())

print(np.log(g))