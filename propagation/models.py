# -*- coding: utf-8 -*-

import tensorflow as tf


class GaussianFields:
    def __init__(self, l, y, mu, W, eps, solver):
        """
        Gaussian Fields method for Knowledge Propagation, as described in [1] (Sect. 11.3)
        [1] Y Bengio et al. - Label Propagation and Quadratic Criterion - Semi-Supervised Learning, MIT Press.

        Energy (cost) function:
            E(f) = ||f_l - y_l||^2 + mu f^T L f + mu eps ||f||^2,

        where El(f) = ||f_l - y_l||^2 encodes the label consistency, and Es(f) = f^T L f + eps f^T f
        encodes the smoothness of the labeling function across the similarity graph.
        :param l: TensorFlow tensor
            N-length {0, 1} integer vector, where L_i = 1 iff the i-th instance is labeled, and 0 otherwise.
        :param y: TensorFlow tensor
            N-length scalar vector, where y_i is the label of the i-th instance.
        :param mu: TensorFlow tensor
            Scalar regularization parameter.
        :param W: TensorFlow tensor
            NxN Adjacency matrix of the similarity graph.
        :param eps: TensorFlow tensor
            Scalar regularization parameter.
        """
        self.l = l
        self.y = y
        self.mu = mu
        self.W = W
        self.eps = eps
        self.solver = solver

    def __call__(self, f):
        """
        Compute the following function:
    
            E(f) = ||f_l - y_l||^2 + mu f^T L f + mu eps ||f||^2,
    
        :param f: TensorFlow tensor
            Vector of N continuous elements.
        :return: TensorFlow tensor
            Energy (cost) of the vector f.
        """
        # Compute the un-normalized graph Laplacian: L = D - W
        W_shp = tf.shape(self.W)

        d = tf.reduce_sum(abs(self.W), axis=-1)
        D = tf.reshape(tf.matrix_diag(d), W_shp)
        L = D - self.W

        # Compute the label consistency
        S = tf.reshape(tf.matrix_diag(self.l), W_shp)
        El = tf.einsum('bm,bmn,bn->b', (f - self.y), S, (f - self.y))

        # Compute the smoothness along the similarity graph
        Es = tf.einsum('bm,bmn,bn->b', f, L, f)

        I = tf.eye(W_shp[-1], batch_shape=[W_shp[0]])
        Er = tf.einsum('bm,bmn,bn->b', f, I, f)

        # Compute the whole cost function
        return El + self.mu * (Es + self.eps * Er)

    def minimize(self):
        """
        Find the global minimum of the following energy (cost) function:

            E(f) = ||f_l - y_l||^2 + mu f^T L f + mu eps ||f||^2,

        Let S denote a diagonal (N x N) matrix, where S_ii = L_i.
        The energy function can be rewritten as:

            E(f) = (f - y)^T S (f - y) + mu f^T L f + mu eps ||f||^2
                 \propto f^T S f - 2 f^T S y + mu (f^T L f + eps f^T I f)
                 = f^T (S + mu L + mu eps I) f - 2 f^T S y.

        The derivative of E(\cdot) w.r.t. f is:

            1/2 d E(f) / d f = (S + mu L + mu eps I) f - S y.

        The second derivative is:

            1/2 d^2 E(f) / df df^T = S + mu L + mu eps I.

        The second derivative is PD when eps > 0 (the graph Laplacian is PSD).
        This ensures that the energy function is minimized when the derivative is set to 0, i.e.:

            (S + mu L + mu eps I) \hat{f} = S y
            \hat{f} = (S + mu L + mu eps I)^-1 S y
        :return: TensorFlow tensor
            Vector \hat{f} that minimizes the energy function E(f).
        """
        # Note: for a justification of L = |D| - W in place of L = D - W, see [2]
        # [2] P Minervini et al. - Discovering Similarity and Dissimilarity Relations for Knowledge Propagation
        #   in Web Ontologies - Journal on Data Semantics, May 2016
        W_shp = tf.shape(self.W)

        d = tf.reduce_sum(abs(self.W), axis=-1)
        D = tf.reshape(tf.matrix_diag(d), W_shp)
        L = D - self.W

        # Compute the coefficient matrix A of the system of linear equations
        S = tf.reshape(tf.matrix_diag(self.l), W_shp)
        I = tf.eye(W_shp[-1], batch_shape=[W_shp[0]])

        A = S + self.mu * (L + self.eps * I)
        b = tf.einsum('bmn,bm->bn', S, self.y)

        hat_f = self.solver.solve(A, tf.expand_dims(b, 2))

        return tf.reshape(hat_f, tf.shape(self.y))
