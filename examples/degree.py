from ggnn.lib import layers, utils

import numpy as np
import argparse

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.data_source_implements import SimpleDataSource
from nnabla.utils.data_iterator import data_iterator
import nnabla.monitor as M
import nnabla.solvers as S

class SimpleDataSource2(SimpleDataSource):
    def __init__(self, xs, shuffle=False):
        def load(position):
            return xs[position]
        super(SimpleDataSource2, self).__init__(load, len(xs), shuffle=shuffle)
    def reset(self):
        super(SimpleDataSource2, self).reset()
        self._position = 0
        if self._shuffle:
            self._rng.shuffle(self._order)

def random_graph(rng):
    num_V = rng.randint(2, 10)
    V = np.ones((num_V, 1))
    E = []
    for i in range(num_V):
        E.append([])
    
    v = list(range(num_V))
    for i in range(num_V):
        rng.shuffle(v)
        n = rng.randint(1, num_V)
        E[i].append((i, "self-loop"))
        for j in v[0:n]:
            if i == j:
                continue
            # add edge (i->j)
            E[i].append((j, "edge"))
    return V, E

def degrees(V, E):
    return list(map(lambda x: len(x) - 1, E))

rng = np.random.RandomState(10000)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-examples", type=int, default=1600)
    parser.add_argument("--num-valid-examples", type=int, default=100)
    parser.add_argument("--accum-grad", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=6400)
    parser.add_argument("--valid-interval", type=int, default=100)

    args = parser.parse_args()

    # prepare dataset
    tdataset = []
    for i in range(args.num_train_examples):
        V, E = random_graph(rng)
        deg = degrees(V, E)
        tdataset.append(([V], [utils.from_neighbor_list(E)], [deg]))

    vdataset = []
    for i in range(args.num_valid_examples):
        V, E = random_graph(rng)
        deg = degrees(V, E)
        vdataset.append(([V], [utils.from_neighbor_list(E)], [deg]))


    # prepare data iterator
    tdata = data_iterator(SimpleDataSource2(tdataset, shuffle=True), 1, False, False, False)
    vdata = data_iterator(SimpleDataSource2(vdataset, shuffle=False), 1, False, False, False)

    # prepare monitors
    monitor = M.Monitor("./degree")
    tloss = M.MonitorSeries("Training Loss", monitor, interval=10)

    verror = M.MonitorSeries("Validation Error", monitor, interval=10)

    # prepare solver
    solver = S.Adam()

    # training loop
    for i in range(args.max_iter):
        l = 0
        for b in range(args.accum_grad):
            # read data
            V, E, degree = tdata.next()
            V = V[0][0]
            E = E[0][0]
            degree = degree[0][0]

            # convert to nn.Variable
            x = nn.Variable(V.shape)
            x.data.data = V
            h_0 = nn.Variable((len(V), 16))
            h_0.data.data = utils.h_0(V, 16)
            label = nn.Variable(degree.shape)
            label.data.data = degree

            # propagate
            h = layers.propagate(h_0, E)

            # output
            output = F.concatenate(h, x)
            output = PF.affine(output, 1)
            label = F.reshape(label, (len(V), 1))
            loss = F.mean(F.squared_error(output, label))

            if i == 0 and b == 0:
                solver.set_parameters(nn.get_parameters())

            # training
            loss.forward(clear_no_need_grad=True)
            loss.backward(clear_buffer=True)
            l += loss.data.data

        solver.update()

        tloss.add(i, l / args.accum_grad)
        l = 0

        if i % args.valid_interval == 0:
            # validation
            # read data
            e = 0
            n = 0
            for b in range(vdata.size):
                V, E, degree = vdata.next()
                V = V[0][0]
                E = E[0][0]
                degree = degree[0][0]

                # convert to nn.Variable
                x = nn.Variable(V.shape)
                x.data.data = V
                h_0 = nn.Variable((len(V), 16))
                h_0.data.data = utils.h_0(V, 16)
                label = nn.Variable(degree.shape)
                label.data.data = degree

                # propagate
                h = layers.propagate(h_0, E)

                # output
                output = F.concatenate(h, x)
                output = PF.affine(output, 1)
                label = F.reshape(label, (len(V), 1))

                error = F.sum(F.less_scalar(F.abs(F.sub2(output, label)), 0.5))

                error.forward()

                e += error.data.data
                n += len(V)
            verror.add(i, e / n)

if __name__ == '__main__':
    train()
