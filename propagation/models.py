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
        
        Definitions:
            - N: number of nodes in the weighted undirected similarity graph.
            - B: batch size.
        
        :param l: [BxN] TensorFlow Tensor.
        :param y: [BxN] TensorFlow Tensor.
        :param mu: [B] TensorFlow Tensor.
        :param W: [BxNxN] TensorFlow Tensor.
        :param eps: [B] TensorFlow Tensor.
        :param solver: propagation.solvers.ASolver instance.
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
    
        :param f: [BxN] TensorFlow tensor
        :return: [B] TensorFlow tensor
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

        r_mu = tf.reshape(self.mu, [-1])
        r_eps = tf.reshape(self.eps, [-1])

        # Compute the whole cost function
        return El + r_mu * (Es + r_eps * Er)

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
        
        Note: for a justification of L = |D| - W in place of L = D - W, see [2]
        [2] P Minervini et al. - Discovering Similarity and Dissimilarity Relations for Knowledge Propagation
            in Web Ontologies - Journal on Data Semantics, May 2016
        
        :return: [BxN] TensorFlow Tensor.
        """
        W_shp = tf.shape(self.W)

        d = tf.reduce_sum(abs(self.W), axis=-1)
        D = tf.reshape(tf.matrix_diag(d), W_shp)
        L = D - self.W

        # Compute the coefficient matrix A of the system of linear equations
        dl = tf.matrix_diag(self.l)
        S = tf.reshape(dl, W_shp)
        I = tf.eye(W_shp[1], batch_shape=[W_shp[0]])

        r_mu = tf.reshape(self.mu, [-1, 1, 1])
        r_mu = tf.tile(r_mu, [1, W_shp[1], W_shp[2]])

        r_eps = tf.reshape(self.eps, [-1, 1, 1])
        r_eps = tf.tile(r_eps, [1, W_shp[1], W_shp[2]])

        A = S + r_mu * (L + r_eps * I)
        b = tf.einsum('bmn,bm->bn', S, self.y)

        b = tf.expand_dims(b, 2)
        hat_f = self.solver.solve(A, b)

        return tf.reshape(hat_f, tf.shape(self.y))
