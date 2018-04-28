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

def get_type_name(s):
    s = s.lower().replace(".", "")
    if s == "wolves":
        return "wolf"
    elif s == "cats":
        return "cat"
    elif s == 'mice':
        return "mouse"
    else:
        return s

class BAbI15DataSource(SimpleDataSource):
    def __init__(self, file, max_samples, shuffle=False):
        dataset = []
        with open(file) as f:
            lines = f.readlines()
            lines = np.array(lines)
            data = lines.reshape((-1, 12))
            for x in data:
                cond = x[0:8]
                question = x[8:12]

                # construct a graph
                V = ["wolf", "mouse", "sheep", "cat"]
                E = []
                for c in cond:
                    words = re.split(r'\s+', c)

                    if len(words) ==  6:
                        # <num> <Name> is a <type>
                        name = words[1]
                        assert(words[2] == 'is')
                        assert(words[3] == 'a')
                        tpe = get_type_name(words[4])

                        if not name in V:
                            V.append(name)

                        E.append((name, tpe, "is"))
                    elif len(words) == 7:
                        # <num> <Type> are afraid of <type>
                        tpe1 = get_type_name(words[1])
                        assert(words[2] == 'are')
                        assert(words[3] == 'afraid')
                        assert(words[4] == 'of')
                        tpe2 = get_type_name(words[5])

                        E.append((tpe1, tpe2, "has_fear"))
                    else:
                        assert(False)

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

                    assert(len(words) == 10)
                    assert(words[1] == 'What')
                    assert(words[2] == 'is')
                    name = words[3]
                    assert(words[4] == 'afraid')
                    assert(words[5] == 'of?')
                    ans = get_type_name(words[6])

                    annotations = np.zeros((len(V), 1))
                    annotations[str2id[name], 0] = 1
                    ans = str2id[ans]
                    dataset.append(([V], [annotations], [E], [ans]))

        dataset = dataset[0:max_samples*4]        

        super(BAbI15DataSource, self).__init__(lambda x: dataset[x], len(dataset), shuffle=shuffle)
    def reset(self):
        super(BAbI15DataSource, self).reset()
        self._position = 0
        if self._shuffle:
            self._rng.shuffle(self._order)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--valid-file", type=str)
    parser.add_argument("--num-training-examples", type=int, default=50)
    parser.add_argument("--accum-grad", type=int, default=5)
    parser.add_argument("--valid-interval", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.95)

    args = parser.parse_args()

    # prepare data iterators
    tdata = data_iterator(BAbI15DataSource(args.train_file, args.num_training_examples, shuffle=True), 1, False, False, False)
    vdata = data_iterator(BAbI15DataSource(args.valid_file, 1000, shuffle=True), 1, False, False, False)

    # prepare monitors
    monitor = M.Monitor("./bAbI15")
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
            h = nn.Variable((len(V), 5))
            h.data.data = utils.h_0(V, 5)
            label = nn.Variable((1, 1))
            label.data.data[0, 0] = ans

            ## propagate
            for _ in range(5):
                h = layers.propagate(h, E)

            ## calculate node score
            s = PF.affine(h, 1)
            s = F.reshape(s, (1, s.shape[0]))

            # initialize solver
            if not solver_initialized:
                solver.set_parameters(nn.get_parameters())
                solver_initialized = True
                solver.zero_grad()

            # calculate loss/error
            s2 = s.unlinked()
            loss = F.mean(F.softmax_cross_entropy(s, label))
            error = F.mean(F.top_n_error(s2, label))
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
                h = nn.Variable((len(V), 5))
                h.data.data = utils.h_0(V, 5)
                label = nn.Variable((1, 1))
                label.data.data[0, 0] = ans

                for _ in range(5):
                    h = layers.propagate(h, E)

                ## calculate node score
                s = PF.affine(h, 1)
                s = F.reshape(s, (1, s.shape[0]))

                # calculate error
                error = F.top_n_error(s, label)
                error.forward(clear_no_need_grad=True)

                if error.data.data > 0.5:
                    if wrong_example is None:
                        wrong_example = (id2str, V, E, ans, s.data.data)
                else:
                    if correct_example is None:
                        correct_example = (id2str, V, E, ans, s.data.data)
                validation_error += error.data.data
            validation_error /= vdata.size
            verror.add(cnt, validation_error)
            accuracy = 1 - validation_error
            if accuracy >= args.threshold:
                def show(example):
                    for i, j in example[2]["is"]:
                        print("{} is {}.".format(example[0][i], example[0][j]))
                    for i, j in example[2]["has_fear"]:
                        print("{} are afraid of {}.".format(example[0][i], example[0][j]))
                    i = np.argmax(example[1])
                    print("What is {} afraid of?".format(example[0][i]))
                    i = np.argmax(example[4])
                    print("Expected: {}, Actual: {}".format(example[0][example[3]], example[0][i]))
                if correct_example is not None:
                    show(correct_example)
                if wrong_example is not None:
                    show(wrong_example)

                break

if __name__ == '__main__':
    train()