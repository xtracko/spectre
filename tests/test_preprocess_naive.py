from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from spectre.sparse.preprocess_naive import rolling_min


class TestRolling(TestCase):
    def test_rolling_min(self):
        dense = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 1, -1, 0, -2, 3, 4, 8],
             [0, 1, 0, 2, 0, 0, 4, 0, 0, 0]])

        csr = csr_matrix(dense)
        csc = csc_matrix(dense)

        self.assertTrue(np.allclose(csr.toarray(), dense))
        self.assertTrue(np.allclose(csc.toarray(), dense))

        self.assertTrue(
            np.allclose(rolling_min(csr, k=1, axis=0).toarray(), dense))
        self.assertTrue(
            np.allclose(rolling_min(csc, k=1, axis=0).toarray(), dense))

        result = np.array([[0, 0, 0, 1, -1, 0, -2, 0, 0, 0]])

        self.assertTrue(
            np.allclose(rolling_min(csr, k=3, axis=0).toarray(), result))
        self.assertTrue(
            np.allclose(rolling_min(csc, k=3, axis=0).toarray(), result))

        result = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, -1, -1, -2, -2, -2, 3],
             [0, 0, 0, 0, 0, 0, 0, 0]])

        self.assertTrue(
            np.allclose(rolling_min(csr, k=3, axis=1).toarray(), result))
        self.assertTrue(
            np.allclose(rolling_min(csc, k=3, axis=1).toarray(), result))

        result = np.array(
            [[0, 0, 0, 1, -1, 0, -2, 3, 4, 8], [0, 0, 0, 1, -1, 0, -2, 0, 0, 0],
             [0, 0, 0, 1, -1, 0, -2, 0, 0, 0]])

        self.assertTrue(np.allclose(
            rolling_min(csr, k=3, axis=0, mode='symmetric').toarray(), result))
        self.assertTrue(np.allclose(
            rolling_min(csc, k=3, axis=0, mode='symmetric').toarray(), result))
