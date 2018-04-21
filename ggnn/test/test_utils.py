import lib.utils as U

import nnabla as nn
import numpy as np
from unittest import TestCase

class TestUtils(TestCase):
    def test_h0(self):
        labels = np.zeros((2, 2))
        labels[0][0] = 1
        labels[1][1] = 1
        x = U.h_0(labels, 4)
        self.assertEqual((2, 4), x.shape)
        self.assertEqual(1, x[0][0])
        self.assertEqual(0, x[0][1])
        self.assertEqual(0, x[0][2])
        self.assertEqual(0, x[0][3])
        self.assertEqual(0, x[1][0])
        self.assertEqual(1, x[1][1])
        self.assertEqual(0, x[1][2])
        self.assertEqual(0, x[1][3])
    def test_from_neighbor_list(self):
        l = [[(1, "B")], [(3, "C")], [(1, "C")], [(2, "B")]]
        expected = { "B": [(0, 1), (3, 2)], "C": [(1, 3), (2, 1)] }
        self.assertEqual(expected, U.from_neighbor_list(l))
def test_from_neighbor_matrix(self):
        m = [[None, "B", None, None], [None, None, None, "C"], [None, "C", None, None], [None, None, "B", None]]
        expected = { "B": [(0, 1), (3, 2)], "C": [(1, 3), (2, 1)] }
        self.assertEqual(expected, U.from_neighbor_matrix(m))
