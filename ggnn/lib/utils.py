import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from .layers import split, stack

def h_0(vertex_annotations, h_size):
    """
    Initilaize vertex state

    Arguments:

    vertex_annotations -- the vertex annotations (numpy.array with shape (|V|, N))
    h_size             -- the size of vertex state

    Return value

    - Return a variable with shape (|V|, h_size)
    """

    h = np.zeros((vertex_annotations.shape[0], h_size))
    h[:, 0:vertex_annotations.shape[1]] = vertex_annotations
    return h

def from_adjacency_list(l):
    """
    Convert adjacency list

    Arguments:

    l -- The adjacency list (the list of (out, label))

    Return value

    - Return a dictionary that represents the graph edge ({label, [in, out]})
    """
    edges = {}
    for i in range(len(l)):
        for j in range(len(l[i])):
            k, label = l[i][j]
            if not (label in edges.keys()):
                edges[label] = []
            edges[label].append((i, k))
    return edges

def from_adjacency_matrix(m):
    """
    Convert adjacency matrix

    Arguments:

    m -- The adjacency matrix (the |V|x|V| matrix (each cell contains a label))

    Return value

    - Return a dictionary that represents the graph edge ({label, [in, out]})
    """

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

def from_edge_list(l):
    """
    Convert edge list

    Arguments:

    l -- The edge list (the list of (in, out, label))

    Return value

    - Return a dictionary that represents the graph edge ({label, [in, out]})
    """
    edges = {}
    for i, j, label in l:
        if not (label in edges.keys()):
            edges[label] = []
        edges[label].append((i, j))
    return edges

def flatten_minibatch(hs, Es):
    """
    Mergem multiple graphs into one graph

    Arguments:

    hs                -- the batched input vertex representations (the list of the numpy array with shape (|V|, D))
    Es                -- the batched dictionary that represents the graph edge (the numpy array of {label, [in, out]})

    Returns value:

    - A tuple of vertex representations and edges
    """

    Vs = np.array(list(map(lambda h: h.shape[0], hs)))
    offsets = np.cumsum(Vs)
    from itertools import chain
    vertices = np.array(list(chain.from_iterable(hs)))

    E = {}
    for i, edges in enumerate(Es):
        if i == 0:
            offset = 0
        else:
            offset = offsets[i - 1]
        for label in edges.keys():
            if not label in E:
                E[label] = []
            E[label].extend(map(lambda x: (x[0] + offset, x[1] + offset), edges[label]))
    return vertices, E
