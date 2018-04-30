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
    def test_from_adjacency_list(self):
        l = [[(1, "B")], [(3, "C")], [(1, "C")], [(2, "B")]]
        expected = { "B": [(0, 1), (3, 2)], "C": [(1, 3), (2, 1)] }
        self.assertEqual(expected, U.from_adjacency_list(l))
    def test_from_adjacency_matrix(self):
        m = [[None, "B", None, None], [None, None, None, "C"], [None, "C", None, None], [None, None, "B", None]]
        expected = { "B": [(0, 1), (3, 2)], "C": [(1, 3), (2, 1)] }
        self.assertEqual(expected, U.from_adjacency_matrix(m))
    def test_from_edge_list(self):
        m = [(0, 1, "B"), (1, 3, "C"), (2, 1, "C"), (3, 2, "B")]
        expected = { "B": [(0, 1), (3, 2)], "C": [(1, 3), (2, 1)] }
        self.assertEqual(expected, U.from_edge_list(m))
    def test_flatten_minibatch(self):
        v1 = np.ones((4, 1))
        e1 = { "B": [(0, 1), (3, 2)], "C": [(2, 1), (1, 3)] }

        v2 = np.ones((2, 1) ) * 2
        e2 = { "B": [(0, 1)] }

        v3 = np.ones((2, 1)) * 3
        e3 = { "C": [(1, 0)]}

        V, E = U.flatten_minibatch([v1, v2, v3], [e1, e2, e3])
        self.assertEqual((8, 1), V.shape)
        self.assertTrue(np.allclose(V[0:4, 0], np.array([1, 1, 1, 1])))
        self.assertTrue(np.allclose(V[4:6, 0], np.array([2, 2])))
        self.assertTrue(np.allclose(V[6:8, 0], np.array([3, 3])))
        self.assertEqual([(0, 1), (3, 2), (4, 5)], E["B"])
        self.assertEqual([(2, 1), (1, 3), (7, 6)], E["C"])
