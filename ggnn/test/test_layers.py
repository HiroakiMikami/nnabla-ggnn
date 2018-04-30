import lib.layers as L

import nnabla as nn
import nnabla.initializer as I
import numpy as np
import math
from unittest import TestCase

class TestLayers(TestCase):
    def test_split(self):
        xs = L.split(nn.Variable((2, 1, 3)))
        self.assertEqual(2, len(xs))
        self.assertEqual((1, 3), xs[0].shape)
        self.assertEqual((1, 3), xs[1].shape)

        xs = L.split(nn.Variable((2, 1, 3)), axis=1)
        self.assertEqual(1, len(xs))
        self.assertEqual((2, 3), xs[0].shape)
    def test_stack(self):
        xs = []

        xs.append(nn.Variable((1, 3)))
        xs.append(nn.Variable((1, 3)))
        self.assertEqual((2, 1, 3), L.stack(xs).shape)

        xs = []
        xs.append(nn.Variable((2, 3)))
        self.assertEqual((2, 1, 3), L.stack(xs, axis=1).shape)
    def test_sum(self):
        xs = []

        xs.append(nn.Variable((1, 3)))
        xs.append(nn.Variable((1, 3)))
        self.assertEqual((), L.sum(xs, axis=None).shape)
        self.assertEqual((1, 3), L.sum(xs, axis=0).shape)

        xs = []
        xs.append(nn.Variable((2, 3)))
        self.assertEqual((), L.sum(xs, axis=None).shape)

    def test_activation(self):
        """

        The graph used bellow is from Fig.1 in the original paper (https://arxiv.org/pdf/1511.05493.pdf)
        """
        edges = { "B": [(0, 1), (3, 2)], "C": [(2, 1), (1, 3)] }

        with nn.parameter_scope("test_shape"):
            vertices = nn.Variable((4, 2))
            outputs = L.activate(vertices, edges, 4)
            params = nn.get_parameters(grad_only=False)

            self.assertEqual((4, 4), outputs.shape)
            self.assertEqual(6, len(params))
            self.assertEqual((2, 4), params["activate_B/affine/W"].shape)
            self.assertEqual((2, 4), params["activate_C/affine/W"].shape)
            self.assertEqual((2, 4), params["activate_reverse_B/affine/W"].shape)
            self.assertEqual((2, 4), params["activate_reverse_C/affine/W"].shape)
            self.assertFalse(params["affine/W"].need_grad)
            self.assertEqual((4, 4), params["affine/W"].shape)
            self.assertEqual((4,), params["affine/b"].shape)
        
        with nn.parameter_scope("test_calc"):
            vertices = nn.Variable((4, 1))
            vertices.data.data = [[1], [2], [3], [4]]

            output = L.activate(vertices, edges)
            params = nn.get_parameters(grad_only=False)
            params["activate_B/affine/W"].data.data = [1.0]
            params["activate_C/affine/W"].data.data = [-0.5]
            params["activate_reverse_B/affine/W"].data.data = [-2]
            params["activate_reverse_C/affine/W"].data.data = [3]
            params["affine/b"].data.data = [10]

            self.assertEqual(1, params["affine/W"].data.data[0, 0])
            output.forward()
            expected = np.zeros((4, 1))
            expected[0][0] = 6
            expected[1][0] = 21.5
            expected[2][0] = 20
            expected[3][0] = 3
            self.assertTrue(np.allclose(expected, output.data.data))
        with nn.parameter_scope("test_isolated_vertex"):
            vertices = nn.Variable((4, 1))
            vertices.data.data = [[1], [2], [3], [4]]

            output = L.activate(vertices, {})
            params = nn.get_parameters(grad_only=False)
            params["affine/b"].data.data = [0]

            output.forward()
            expected = np.zeros((4, 1))
            expected[:,: ] = 0
            self.assertTrue(np.allclose(expected, output.data.data))

    def test_propagate(self):
        """

        The graph used bellow is from Fig.1 in the original paper (https://arxiv.org/pdf/1511.05493.pdf)
        """
        edges = { "B": [(0, 1), (3, 2)], "C": [(2, 1), (1, 3)] }

        with nn.parameter_scope("test_propagate"):
            vertices = nn.Variable((4, 1))
            outputs = L.propagate(vertices, edges)
            params = nn.get_parameters()

            self.assertEqual((4, 1), outputs.shape)
            self.assertEqual(8, len(params))
            self.assertEqual((1, 3, 1), params["W_zr/affine/W"].shape)
            self.assertEqual((1, 2, 1), params["U_zr/affine/W"].shape)
            self.assertEqual((1, 1), params["U/affine/W"].shape)
    
    def test_node_representation(self):
        h = nn.Variable((1, 1))
        h.data.data[0] = 1
        x = nn.Variable((1, 1))
        x.data.data[0] = 2

        with nn.parameter_scope("test_node_representation"):
            r = L.node_representation(h, x, 1, w_init=I.ConstantInitializer(1), b_init=I.ConstantInitializer(0))
            self.assertEqual((1, 1), r.shape)
            r.forward()
            self.assertEqual(3, r.data.data[0, 0])

    def test_graph_representation(self):
        h = nn.Variable((2, 1))
        h.data.data[0] = 1
        h.data.data[1] = 2
        x = nn.Variable((2, 1))
        x.data.data[0] = 2
        x.data.data[1] = 4

        with nn.parameter_scope("test_graph_representation"):
            r = L.graph_representation(h, x, 1, w_init=I.ConstantInitializer(1), b_init=I.ConstantInitializer(0))
            self.assertEqual((1, 1), r.shape)
            r.forward()

            actual = 0
            actual += 1 / (1 + math.exp(-3)) * math.tanh(3)
            actual += 1 / (1 + math.exp(-6)) * math.tanh(6)
            self.assertTrue(np.allclose(actual, r.data.data[0, 0]))


    def node_annotation(self):
        h = nn.Variable((1, 2))
        h.data.data[0, 0] = 1
        h.data.data[0, 1] = 0
        x = nn.Variable((1, 1))
        x.data.data[0] = 2

        with nn.parameter_scope("test_node_annotation"):
            h, x = L.node_annotation(h, x, 1, w_init=I.ConstantInitializer(1), b_init=I.ConstantInitializer(0))
            self.assertEqual((1, 2), h.shape)
            self.assertEqual((1, 1), x.shape)
            F.sink(h, x).forward()

            actual = 1 / (1 + math.exp(-3))
            self.assertTrue(np.allclose(actual, h.data.data[0, 0]))
            self.assertEqual(0, h.data.data[0, 1])
            self.assertTrue(np.allclose(actual, x.data.data[0, 0]))
