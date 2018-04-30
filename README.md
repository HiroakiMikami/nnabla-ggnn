nnabla-ggnn: NNabla Implementation of GG-NN
===

This repository is a NNabla implementation of Gated Graph Sequence Neural Networks (GG-NN) proposed in the paper [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) by Y.Li, D.Tarlwo, M.Brockschmdit, and R. Zemel. GG-NNs can use graph-structured data as inputs of neural networks, and gets high accuracy on some bAbI-tasks. This implementation is tested with bAbI 15 and bAbI 19, and gets high accuracy (100% for bAbI15 and 95% for bAbI 19).
The official implementation is available in the [GitHub repository](https://github.com/yujiali/ggnn).

Requirements
---

* Python 3.x (tested with Python 3.6.5)
* NNabla 0.9.9

Run Examples
---

### bAbI 15

```bash
$ babi-tasks 15 1000 > train.txt # Notes: babi-tasks can be installed from https://github.com/facebook/bAbI-tasks
$ babi-tasks 15 1000 > vaild.txt
$ python ./main.py bAbI15 --train-file train.txt --valid-file valid.txt # --context cudnn
```

My result: get 100% validation accuracy after 200 iterations (1 epochs).

### bAbI 19

```bash
$ babi-tasks 19 1000 > train.txt # Notes: babi-tasks can be installed from https://github.com/facebook/bAbI-tasks
$ babi-tasks 19 1000 > vaild.txt
$ python ./main.py bAbI10 --train-file train.txt --valid-file valid.txt # --context cudnn
```

My result: get 95% validation accuracy after 54000 iterations (216 epochs).


TODO
---

* [ ] mini-batched training
    * I didn't implement the mini-batch version.
    * I think it is not difficult to implement mini-batched training if the number of vertices in graph is same.

References
---

* [Gated Graph Sequence Neural Networks, ICLR 2016](https://arxiv.org/abs/1511.05493)
* [the official GitHub repository](https://github.com/yujiali/ggnn)
* [bAbI-tasks](https://github.com/facebook/bAbI-tasks)
