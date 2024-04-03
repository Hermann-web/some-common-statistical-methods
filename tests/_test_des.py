import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

nb_exp = 1000

# full paralel
Liste_test = np.random.uniform(1, 6, (6, nb_exp)).sum(axis=0)
# Liste_test = np.random.binomial(4,0.7,(6,nb_exp)).mean(axis=0)
# Liste_test = np.random.normal(4,0.7,(6,nb_exp)).mean(axis=0)

# semi-paralel: binomial test
# Liste_test = [np.random.binomial(4,0.7,6).sum(axis=0) for _ in range(nb_exp)]
# Liste_test = np.random.binomial(4,0.7,(6,nb_exp)).mean(axis=0)

# test normality
# print(ME_Normal_dist(Liste_test, debug=True))

# plt.hist(Liste_test, bins=100)
ax = sns.distplot(Liste_test)
plt.show()
