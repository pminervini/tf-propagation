# tf-propagation [![Build Status](https://travis-ci.org/pminervini/tf-propagation.svg?branch=master)](https://travis-ci.org/pminervini/tf-propagation)

End-to-en differentiable Label Propagation library, implemented using TensorFlow.

The underlying idea is that it is possible to spread labels from one (or a few nodes)
in an undirected graph to all `n` nodes, by minimising the following cost function,
defined over a node labeling `f \in R^n`:

```text
E(f) = \sum_{i \in L} (f[i] - y[i])^2 + mu \sum_i \sum_j W_ij(f[i] - f[j])^2 + mu eps ||f||^2.
```

The term `\sum_{i \in L} (f[i] - y[i])^2` enforces consistency of labeled nodes in `L`
with a gold labeling `y`.
The term `mu \sum_i \sum_j W_ij(f[i] - f[j])^2` enforces that, given two nodes that are
connected in the undirected graph (i.e. `W_ij = W_ji > 0`), they are associated to a similar labeling.
The last term is a L2 regulariser.

Since the cost function `E(f)` is quadratic, it has one closed-form solution for `mu > 0` and `eps > 0`.
Furthermore, it is possible to *backpropagate* the error resulting from the propagation process, back to the graph creation.

### Examples

Here's a small demo where we propagate the labels (+1.1 and -1.0) from two nodes (upper left and lower right) to all nodes in an undirected graph structured as a `40 x 40` grid:

![Demo](http://data.neuralnoise.com/tf-propagation/demo.png)

Here we learn the optimal edge weights by minimizing the prediction error via gradient descent:

![AdaptiveDemo](http://data.neuralnoise.com/tf-propagation/tty.gif)

### References

We refer to [TWEB], and [JoDS] for a full proof derivation, a complexity analysis, and more information on learning the optimal similarity graph from data using backprop.

[TWEB]: [Adaptive Knowledge Propagation in Web Ontologies](https://dl.acm.org/citation.cfm?id=3105961). TWEB 12(1): 2:1-2:28 (2018)

[JoDS]: [Discovering Similarity and Dissimilarity Relations for Knowledge Propagation in Web Ontologies](https://link.springer.com/article/10.1007/s13740-016-0062-7). J. Data Semantics 5(4): 229-248 (2016)

### BibTex

```text
@article{DBLP:journals/tweb/MinerviniTdF18,
  author    = {Pasquale Minervini and
               Volker Tresp and
               Claudia d'Amato and
               Nicola Fanizzi},
  title     = {Adaptive Knowledge Propagation in Web Ontologies},
  journal   = {{TWEB}},
  volume    = {12},
  number    = {1},
  pages     = {2:1--2:28},
  year      = {2018},
  url       = {http://doi.acm.org/10.1145/3105961},
  doi       = {10.1145/3105961}
}

@article{DBLP:journals/jodsn/MinervinidFT16,
  author    = {Pasquale Minervini and
               Claudia d'Amato and
               Nicola Fanizzi and
               Volker Tresp},
  title     = {Discovering Similarity and Dissimilarity Relations for Knowledge Propagation
               in Web Ontologies},
  journal   = {J. Data Semantics},
  volume    = {5},
  number    = {4},
  pages     = {229--248},
  year      = {2016},
  url       = {https://doi.org/10.1007/s13740-016-0062-7},
  doi       = {10.1007/s13740-016-0062-7}
}
```
