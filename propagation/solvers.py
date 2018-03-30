# -*- coding: utf-8 -*-

import abc

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


class ASolver(abc.ABC):
    @abc.abstractmethod
    def solve(self, A, b):
        raise NotImplementedError("Solver not implemented")


class ExactSolver(ASolver):
    def solve(self, A, B):
        """
        Solve a system of linear equations.

        Finds X such that:
            A X = B.

        :param A: [BxMxM] TensorFlow Tensor.
        :param B: [BxMx1] TensorFlow Tensor.
        :return: X: [BxMx1] TensorFlow Tensor.
        """
        return tf.matrix_solve_ls(matrix=A, rhs=B)


class JacobiSolver(ASolver):
    def __init__(self, nb_iterations=10):
        self.nb_iterations = nb_iterations

    def solve(self, A, B):
        """
        Solve a system of linear equations using the Jacobi method.
        https://en.wikipedia.org/wiki/Jacobi_method

        Finds X such that:
            A X = B.

        :param A: [BxMxM] TensorFlow Tensor.
        :param B: [BxMx1] TensorFlow Tensor.
        :return: X: [BxMx1] TensorFlow Tensor.
        """
        d = tf.matrix_diag_part(A)
        D = tf.reshape(tf.matrix_diag(d), tf.shape(A))
        R = A - D

        iD = tf.reshape(tf.matrix_diag(1.0 / d), tf.shape(A))

        X = tf.zeros_like(B)
        for _ in range(self.nb_iterations):
            T = tf.einsum('bmn,bno->bmo', R, X)
            S = B - T
            X = tf.einsum('bmn,bno->bmo', iD, S)
        return tf.reshape(X, tf.shape(B))
