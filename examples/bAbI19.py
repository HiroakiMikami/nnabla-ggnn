from ggnn.lib import layers, utils

import re
import numpy as np
import argparse

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.data_source_implements import SimpleDataSource
from nnabla.utils.data_iterator import data_iterator
import nnabla.monitor as M
import nnabla.solvers as S

id2classes = ['s', 'n', 'w', 'e', 'end']
classes = { 's': 0, 'n': 1, 'w': 2, 'e': 3, 'end': 4 }

class BAbI19DataSource(SimpleDataSource):
    def __init__(self, file, max_samples, shuffle=False):
        dataset = []
        with open(file) as f:
            lines = f.readlines()
            lines = np.array(lines)
            data = lines.reshape((-1, 5))
            for x in data:
                cond = x[0:4]
                question = x[4:5]

                # construct a graph
                V = []
                E = []
                for c in cond:
                    words = re.split(r'\s+' , c)
                    assert(len(words) ==  8)
                    # <num> The <X> is <dir> the <Y>
                    assert(words[1] == 'The')
                    X = words[2]
                    assert(words[3] == 'is')
                    dir = words[4][0]
                    assert(words[5] == 'the')
                    Y = words[6].replace('.', '')

                    if not X in V:
                        V.append(X)
                    if not Y in V:
                        V.append(Y)

                    E.append((X, Y, dir))

                # convert name to index
                V2 = list(enumerate(V))
                str2id = {}
                for id, name in V2:
                    str2id[name] = id
                E2 = []
                for (s1, s2, label) in E:
                    E2.append((str2id[s1], str2id[s2], label))
                E = utils.from_edge_list(E2)

                # generate a dataset for each question
                for q in question:
                    words = re.split(r'\s+', q)

                    assert(len(words) == 13)
                    assert(words[1] == 'What')
                    assert(words[2] == 'is')
                    assert(words[3] == 'the')
                    assert(words[4] == 'path')
                    assert(words[5] == 'from')
                    X = words[6]
                    assert(words[7] == 'to')
                    Y = words[8].replace('?', '')
                    ans = words[9].split(',')

                    annotations = np.zeros((len(V), 2))
                    annotations[str2id[X], 0] = 1
                    annotations[str2id[Y], 1] = 1
                    label = []
                    for a in ans:
                        label.append(classes[a])
                    label.append(classes['end'])
                    dataset.append(([V], [annotations], [E], [label]))

        dataset = dataset[0:max_samples]

        super(BAbI19DataSource, self).__init__(lambda x: dataset[x], len(dataset), shuffle=shuffle)
    def reset(self):
        super(BAbI19DataSource, self).reset()
        self._position = 0
        if self._shuffle:
            self._rng.shuffle(self._order)

def step(h, x, E):
    for _ in range(5):
        h = layers.propagate(h, E)

    # output
    with nn.parameter_scope("output"):
        output = layers.graph_representation(h, x, len(classes))

    # annotation
    with nn.parameter_scope("annotation"):
        h, x = layers.node_annotaiton(h, x)
    return h, x, output

def predict(V, E, n):
    # convert to nn.Variable
    x = nn.Variable(V.shape)
    x.data.data = V
    h = nn.Variable((len(V), 6))
    h.data.data = utils.h_0(V, 6)

    outputs = []
    for _ in range(n):
        # propagate
        h, x, output = step(h, x, E)
        outputs.append(output)

    return outputs

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--valid-file", type=str)
    parser.add_argument("--num-training-examples", type=int, default=250)
    parser.add_argument("--accum-grad", type=int, default=1)
    parser.add_argument("--valid-interval", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--context", type=str, default="cpu")
    parser.add_argument("--device-id", type=int, default=0)

    args = parser.parse_args()

    from nnabla.ext_utils import get_extension_context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # prepare data iterators
    tdata = data_iterator(BAbI19DataSource(args.train_file, args.num_training_examples, shuffle=True), 1, False, False, False)
    vdata = data_iterator(BAbI19DataSource(args.valid_file, 1000, shuffle=True), 1, False, False, False)

    # prepare monitors
    monitor = M.Monitor("./bAbI19")
    tloss = M.MonitorSeries("Training Loss", monitor, interval=10)
    terror = M.MonitorSeries("Training Error", monitor, interval=10)
    verror = M.MonitorSeries("Validation Error", monitor, interval=1)

    # prepare solver
    solver = S.Adam()
    solver_initialized = False

    cnt = 0
    while True:
        l = 0.0
        e = 0.0

        solver.zero_grad()
        for _ in range(args.accum_grad):
            # read next data
            x = tdata.next()
            V = x[1][0][0]
            E = x[2][0][0]
            ans = x[3][0][0]

            # construct GGNN
            ## convert to nn.Variable
            x = nn.Variable(V.shape)
            x.data.data = V
            h = nn.Variable((len(V), 6))
            h.data.data = utils.h_0(V, 6)

            outputs = predict(V, E, len(ans))
            losses = []
            errors = []
            for a, output in zip(ans, outputs):
                label = nn.Variable((1, 1))
                label.data.data[0, 0] = a

                losses.append(F.mean(F.softmax_cross_entropy(output, label)))
                output2 = output.unlinked()
                errors.append(F.mean(F.top_n_error(output2, label)))

            # initialize solver
            if not solver_initialized:
                solver.set_parameters(nn.get_parameters())
                solver_initialized = True
                solver.zero_grad()

            # calculate loss/error
            loss = F.mean(F.stack(*losses))
            error = F.mean(F.stack(*errors))
            F.sink(loss, error).forward(clear_no_need_grad=True)
            loss.backward(clear_buffer=True)

            l += loss.data.data
            e += error.data.data

        # dump log
        tloss.add(cnt, l / args.accum_grad)
        terror.add(cnt, e / args.accum_grad)
        l = 0.0
        e = 0.0

        solver.update()

        cnt += 1
        if cnt % args.valid_interval == 0:
            # validation
            validation_error = 0
            correct_example = None
            wrong_example = None
            for _ in range(vdata.size):
                x = vdata.next()
                id2str = x[0][0][0]
                V = x[1][0][0]
                E = x[2][0][0]
                ans = x[3][0][0]

                # construct GGNN
                ## convert to nn.Variable
                x = nn.Variable(V.shape)
                x.data.data = V
                h = nn.Variable((len(V), 6))
                h.data.data = utils.h_0(V, 6)

                outputs = predict(V, E, len(ans))
                errors = []
                actual = []
                for a, output in zip(ans, outputs):
                    label = nn.Variable((1, 1))
                    label.data.data[0, 0] = a

                    errors.append(F.mean(F.top_n_error(output, label)))
                    actual.append(output.data.data)

                error = F.mean(F.stack(*errors))
                error.forward(clear_no_need_grad=True)

                x = 0.0
                if error.data.data == 0:
                    x = 0
                else:
                    x = 1

                if x > 0.5:
                    if wrong_example is None:
                        wrong_example = (id2str, V, E, ans, actual)
                else:
                    if correct_example is None:
                        correct_example = (id2str, V, E, ans, actual)
                validation_error += x
            validation_error /= vdata.size
            verror.add(cnt, validation_error)
            accuracy = 1 - validation_error
            if accuracy >= args.threshold:
                def show(example):
                    if "s" in example[2]:
                        for i, j in example[2]["s"]:
                            print("The {} is south the {}.".format(example[0][i], example[0][j]))
                    if "n" in example[2]:
                        for i, j in example[2]["n"]:
                            print("The {} is north the {}.".format(example[0][i], example[0][j]))
                    if "w" in example[2]:
                        for i, j in example[2]["w"]:
                            print("The {} is west the {}.".format(example[0][i], example[0][j]))
                    if "e" in example[2]:
                        for i, j in example[2]["e"]:
                            print("The {} is east the {}.".format(example[0][i], example[0][j]))
                    i = np.argmax(example[1][:, 0])
                    j = np.argmax(example[1][:, 1])
                    print("What is the path from {} to {}?".format(example[0][i], example[0][j]))

                    for (expected, actual) in zip(example[3], example[4]):
                        i = np.argmax(actual[0])
                        print("Expected: {}, Actual: {}".format(id2classes[expected], id2classes[i]))
                if correct_example is not None:
                    show(correct_example)
                if wrong_example is not None:
                    show(wrong_example)

                break

if __name__ == '__main__':
    train()