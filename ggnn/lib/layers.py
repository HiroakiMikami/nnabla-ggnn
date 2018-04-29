import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

def split(x, axis=0):
    if x.shape[axis] == 1:
        s = list(x.shape)
        s.pop(axis)
        x = F.broadcast(x, x.shape)
        return [F.reshape(x, s)]
    else:
        return F.split(x, axis=axis)

def stack(xs, axis=0):
    if len(xs) == 1:
        s = list(xs[0].shape)
        s.insert(axis, 1)
        xs[0] = F.broadcast(xs[0], xs[0].shape)
        return F.reshape(xs[0], s)
    else:
        return F.stack(*xs, axis=axis)

def sum(xs, axis=None):
    return F.sum(stack(xs), axis=axis)

def activate(h, edges, state_size=None, bias_initializer=None, edge_initializers = None):
    """
    Activate vertex representations

    Arguments:

    h                 -- the input vertex representations (nnabla.Variable with shape (|V|, D))
    edges             -- the dictionary that represents the graph edge ({label, [in, out]})
    state_size        -- (optional) the size of hidden state (h.shape[1] is used if this argument is None)
    bias_initializer  -- (optional) the parameter initializer for bias
    edge_initializers -- (optional) the parameter initializers ({label, (Initializer, Initializer)})

    Return value

    - Return a variable with shape (|V|, state_size)
    """
    if state_size is None:
        state_size = h.shape[1]

    # split vertex representations to access each representation
    vertices = split(h)
    number_of_vertices = len(vertices)

    state = []
    for _ in range(number_of_vertices):
        state.append([])
    for label in edges.keys():
        es = edges[label]

        # forward edges
        input = stack(list(map(lambda x: vertices[x[0]], es)))
        with nn.parameter_scope("activate_{}".format(label)):
            w_init = None
            if edge_initializers is not None:
                w_init = edge_initializers[label][0]
            os = PF.affine(input, state_size, w_init = w_init, with_bias = False)
        os = split(os)
        for edge, output in zip(es, os):
            state[edge[1]].append(output)

        # reverse edges
        input = stack(list(map(lambda x: vertices[x[1]], es)))
        with nn.parameter_scope("activate_reverse_{}".format(label)):
            w_init = None
            if edge_initializers is not None:
                w_init = edge_initializers[label][1]
            os = PF.affine(input, state_size, w_init = w_init, with_bias = False)
        os = split(os)
        for edge, output in zip(es, os):
            state[edge[0]].append(output)

    # assemble state
    outputs = stack(list(map(lambda x: sum(x, axis=0), state)))

    # add bias
    W = nn.parameter.get_parameter_or_create("affine/W", (state_size, state_size), need_grad=False)
    # initialize
    W.data.data = np.identity(state_size, dtype=np.float32)
    outputs = PF.affine(outputs, state_size, with_bias=True, b_init=bias_initializer)

    return outputs

def propagate(h, edges, state_size=None,
              w_initializer=None, u_initializer1=None, u_initializer2=None,
              bias_initializer=None, edge_initializers = None):
    """
    Propagate vertex representations

    Arguments:

    h                 -- the input vertex representations (nnabla.Variable with shape (|V|, D))
    edges             -- the dictionary that represents the graph edge ({label, [in, out]})
    state_size        -- (optional) the size of hidden state (h.shape[1] is used if this argument is None)
    w_initializer     -- (optional)
    u_initializer1    -- (optional)
    u_initializer2    -- (optional)
    bias_initializer  -- (optional)
    edge_initializers -- (optional)

    Return value

    - Return a variable with shape (|V|, D)
    """
    if state_size is None:
        state_size = h.shape[1]
    h_size = h.shape[1]
    with nn.parameter_scope("activate"):
        a = activate(h, edges, state_size, bias_initializer=bias_initializer, edge_initializers=edge_initializers)
    with nn.parameter_scope("W_zr"):
        ws = PF.affine(a, (3, h_size), with_bias=False, w_init=w_initializer)
    (z1, r1, h_hat1) = split(ws, axis=1)
    with nn.parameter_scope("U_zr"):
        us = PF.affine(h, (2, state_size), with_bias=False, w_init=u_initializer1)
    (z2, r2) = split(us, axis=1)
    z = F.sigmoid(F.add2(z1, z2))
    r = F.sigmoid(F.add2(r1, r2))
    with nn.parameter_scope("U"):
        h_hat2 = PF.affine(F.mul2(r, h), state_size, with_bias=False, w_init=u_initializer2)
    h_hat = F.tanh(F.add2(h_hat1, h_hat2))
    return F.add2(F.sub2(h, F.mul2(z, h)), F.mul2(z, h_hat))

def node_representation(h, x, n_outmaps, w_init=None, b_init=None):
    """
    Outputs node selection/representaiton model

    Arguments:

    h                 -- the input vertex representations (nnabla.Variable with shape (|V|, H))
    x                 -- the input vertex annotation (nnabla.Variable with shape (|V|, X))
    n_outmaps         -- the size of node representation
    w_init            -- (optional)
    b_init            -- (optional)

    Return value

    - Return a variable with shape (|V|, n_outmaps)
    """
    with nn.parameter_scope("node_representation"):
        return PF.affine(F.concatenate(h, x), n_outmaps, w_init=w_init, b_init=b_init)

def graph_representation(h, x, n_outmaps, w_init=None, b_init=None):
    """
    Outputs graph representaiton model

    Arguments:

    h                 -- the input vertex representations (nnabla.Variable with shape (|V|, H))
    x                 -- the input vertex annotation (nnabla.Variable with shape (|V|, X))
    n_outmaps         -- the size of node representation
    w_init            -- (optional)
    b_init            -- (optional)

    Return value

    - Return a variable with shape (n_outmaps)
    """
    with nn.parameter_scope("graph_representation"):
        output = F.concatenate(h, x)
        output = PF.affine(output, (2, n_outmaps), w_init=w_init, b_init=b_init)
        (s, t) = F.split(output, axis=1)
        return F.sum(F.mul2(F.sigmoid(s), F.tanh(t)), axis=0, keepdims=True)

def node_annotaiton(h, x, w_init=None, b_init=None):
    """
    Outputs graph annotation model

    Arguments:

    h                 -- the input vertex representations (nnabla.Variable with shape (|V|, H))
    x                 -- the input vertex annotation (nnabla.Variable with shape (|V|, X))
    w_init            -- (optional)
    b_init            -- (optional)

    Return value

    - Return a tuple of vertex representations and annotation for a next step
    """
    with nn.parameter_scope("node_annotation"):
        s = F.concatenate(h, x)
        x = F.sigmoid(PF.affine(s, x.shape[1], w_init=w_init, b_init=b_init))
        z = nn.Variable((x.shape[0], h.shape[1] - x.shape[1]))
        z.data.data[:, :] = 0.0
        h = F.concatenate(x, z)
        return h, x
