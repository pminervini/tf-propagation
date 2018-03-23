# -*- coding: utf-8 -*-

import abc

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


class ASolver(abc.ABC):
    @abc.abstractmethod
    def solve(self, A, B):
        raise NotImplementedError("Solver not implemented")


class ExactSolver(ASolver):
    def solve(self, A, B):
        return tf.matrix_solve_ls(matrix=A, rhs=B)


class JacobiSolver(ASolver):
    def __init__(self, nb_iterations=10):
        self.nb_iterations = nb_iterations

    def solve(self, A, B):
        d = tf.diag_part(A)
        D = tf.diag(d)
        R = A - D

        iD = tf.diag(1.0 / d)

        X = tf.zeros_like(B)
        for _ in range(self.nb_iterations):
            T = tf.tensordot(R, X, axes=[1, 0])
            S = B - T
            X = tf.tensordot(iD, S, axes=[1, 0])
        return X
