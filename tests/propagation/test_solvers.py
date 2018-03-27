# -*- coding: utf-8 -*-

import pytest

import numpy as np
import tensorflow as tf

from propagation.solvers import ASolver, ExactSolver, JacobiSolver


class NPSolver(ASolver):
    def __init__(self, nb_iterations=10):
        self.nb_iterations = nb_iterations

    def solve(self, A, B):
        dA = np.diag(A)
        D = np.diag(dA)
        R = A - D

        iD = np.diag(1.0 / dA)

        X = np.zeros_like(B)

        for _ in range(self.nb_iterations):
            # T = np.dot(R, X)
            # S = B - T
            # X = iD.dot(S)

            T = np.tensordot(R, X, axes=1)
            S = B - T
            X = np.tensordot(iD, S, axes=1)

        return X


def test_solvers():
    A_ph = tf.placeholder('float32', [None, None], name='A')
    B_ph = tf.placeholder('float32', [None, None], name='B')

    es = ExactSolver()
    js = JacobiSolver(nb_iterations=100)
    ns = NPSolver(nb_iterations=100)

    eX = es.solve(A=A_ph, B=B_ph)
    jX = js.solve(A=A_ph, B=B_ph)

    A_value = np.array([[4.0, -1.0, 1.0],
                        [4.0, -8.0, 1.0],
                        [-2.0, 1.0, 5.0]])

    B_value = np.array([[7.0], [-21.0], [15.0]])

    with tf.Session() as session:
        feed_dict = {A_ph: A_value, B_ph: B_value}
        eX_value = session.run(eX, feed_dict=feed_dict)
        jX_value = session.run(jX, feed_dict=feed_dict)
        nX_value = ns.solve(A=A_value, B=B_value)

        eAX_value = np.dot(A_value, eX_value)
        jAX_value = np.dot(A_value, jX_value)
        nAX_value = np.dot(A_value, nX_value)

        np.testing.assert_allclose(eAX_value, B_value, rtol=0.001)
        np.testing.assert_allclose(jAX_value, B_value, rtol=0.001)
        np.testing.assert_allclose(nAX_value, B_value, rtol=0.001)

        np.testing.assert_allclose(eX_value, jX_value, rtol=0.001)
        np.testing.assert_allclose(eX_value, nX_value, rtol=0.001)

if __name__ == '__main__':
    pytest.main([__file__])
