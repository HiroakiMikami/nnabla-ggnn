import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

def h_0(vertex_labels, h_size):
    h = np.zeros((vertex_labels.shape[0], h_size))
    h[:, 0:vertex_labels.shape[1]] = vertex_labels
    return h

def from_neighbor_list(l):
    edges = {}
    for i in range(len(l)):
        for j in range(len(l[i])):
            k, label = l[i][j]
            if not (label in edges.keys()):
                edges[label] = []
            edges[label].append((i, k))
    return edges

def from_neighbor_matrix(m):
    edges = {}
    for i in range(len(m)):
        for j in range(len(m)):
            label = m[i][j]
            if label is None:
                continue
            if not (label in edges.keys()):
                edges[label] = []
            edges[label].append((i, j))
    return edges