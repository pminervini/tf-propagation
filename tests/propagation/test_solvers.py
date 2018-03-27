# -*- coding: utf-8 -*-

import pytest

import numpy as np
import tensorflow as tf

from propagation.solvers import ExactSolver, JacobiSolver


def test_batch_solvers():
    A_ph = tf.placeholder('float32', [1, None, None], name='A')
    B_ph = tf.placeholder('float32', [1, None, None], name='B')

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

if __name__ == '__main__':
    pytest.main([__file__])


