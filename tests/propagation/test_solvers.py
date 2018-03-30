# -*- coding: utf-8 -*-

import pytest

import numpy as np
import tensorflow as tf

from propagation.solvers import ExactSolver, JacobiSolver


def test_batch_solvers():
    A_ph = tf.placeholder('float32', [None, None, None], name='A')
    B_ph = tf.placeholder('float32', [None, None, None], name='B')

    es = ExactSolver()
    js = JacobiSolver(nb_iterations=10)

    eX = es.solve(A=A_ph, B=B_ph)
    jX = js.solve(A=A_ph, B=B_ph)

    A_value = np.array([[4.0, -1.0, 1.0],
                        [4.0, -8.0, 1.0],
                        [-2.0, 1.0, 5.0]]).reshape(1, 3, 3)

    B_value = np.array([7.0, -21.0, 15.0]).reshape(1, 3, 1)

    with tf.Session() as session:
        feed_dict = {A_ph: A_value, B_ph: B_value}

        eX_value = session.run(eX, feed_dict=feed_dict)
        jX_value = session.run(jX, feed_dict=feed_dict)

        np.testing.assert_allclose(eX_value, jX_value, rtol=1e-4)


def test_batch_solvers_2():
    A_ph = tf.placeholder('float32', [None, None, None], name='A')
    B_ph = tf.placeholder('float32', [None, None, None], name='B')

    es = ExactSolver()
    js = JacobiSolver(nb_iterations=10)

    eX = es.solve(A=A_ph, B=B_ph)
    jX = js.solve(A=A_ph, B=B_ph)

    A_value = np.array([[4.0, -1.0, 1.0],
                        [4.0, -8.0, 1.0],
                        [-2.0, 1.0, 5.0]]).reshape(3, 3)

    B_value = np.array([7.0, -21.0, 15.0]).reshape(3, 1)

    batch_A_value = np.zeros(shape=[2, 3, 3], dtype='float32')
    batch_B_value = np.zeros(shape=[2, 3, 1], dtype='float32')

    batch_A_value[0, :, :] = A_value
    batch_B_value[0, :, :] = B_value

    batch_A_value[1, :, :] = A_value
    batch_B_value[1, :, :] = B_value

    with tf.Session() as session:
        feed_dict = {A_ph: batch_A_value, B_ph: batch_B_value}

        eX_value = session.run(eX, feed_dict=feed_dict)
        jX_value = session.run(jX, feed_dict=feed_dict)

        np.testing.assert_allclose(eX_value, jX_value, rtol=1e-4)


def test_batch_solvers_3():
    A_ph = tf.placeholder('float32', [None, None, None], name='A')
    B_ph = tf.placeholder('float32', [None, None, None], name='B')

    es = ExactSolver()
    js = JacobiSolver(nb_iterations=10)

    eX = es.solve(A=A_ph, B=B_ph)
    jX = js.solve(A=A_ph, B=B_ph)

    A_value = np.array([[4.0, -1.0, 1.0],
                        [4.0, -8.0, 1.0],
                        [-2.0, 1.0, 5.0]]).reshape(3, 3)

    B_value = np.array([[7.0, -21.0, 15.0],
                        [7.0, -21.0, 15.0]]).reshape(3, 2)

    batch_A_value = np.zeros(shape=[2, 3, 3], dtype='float32')
    batch_B_value = np.zeros(shape=[2, 3, 2], dtype='float32')

    batch_A_value[0, :, :] = A_value
    batch_B_value[0, :, :] = B_value

    batch_A_value[1, :, :] = A_value
    batch_B_value[1, :, :] = B_value

    with tf.Session() as session:
        feed_dict = {A_ph: batch_A_value, B_ph: batch_B_value}

        eX_value = session.run(eX, feed_dict=feed_dict)
        jX_value = session.run(jX, feed_dict=feed_dict)

        np.testing.assert_allclose(eX_value, jX_value, rtol=1e-4)

if __name__ == '__main__':
    pytest.main([__file__])
