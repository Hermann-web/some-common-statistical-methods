import unittest

import numpy as np

from statanalysis.utils_md.preprocessing import clear_mat_vec


class Tests_hp_estimators_regression(unittest.TestCase):

    def test_fisher(self):
        A = np.array([[1, 3], [4, 3], [5, 3], [7, np.nan]])
        y = np.array([6, np.nan, 3, 2])
        A1, y1 = clear_mat_vec(A, y)
        print("A1: ", A1)
        print("y: ", y1)
        assert (A1 - np.array([
            [1, 3],
            # [4,3],
            [5, 3],
            # [7,np.nan]
        ])).any() == False
        assert (y1 - np.array([6, 3])).any() == False


if __name__ == "__main__":
    unittest.main()
