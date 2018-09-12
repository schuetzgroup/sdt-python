import numpy as np

from sdt import funcs


def test_step_function():
    x = np.arange(20)

    # left-sided
    f = funcs.StepFunction(x, x)
    np.testing.assert_allclose(f([-1, 1, 2.3, 2.5, 2.8, 3, 3.7, 100]),
                               [0, 1, 3, 3, 3, 3, 4, 19])

    # right-sided
    f = funcs.StepFunction(x, x, side="right")
    np.testing.assert_allclose(f([-1, 1, 2.3, 2.5, 2.8, 3, 3.7, 100]),
                               [0, 1, 2, 2, 2, 3, 3, 19])

    # sorting
    f = funcs.StepFunction(x[::-1], x)
    np.testing.assert_allclose(f.x, x)
    np.testing.assert_allclose(f.y, x[::-1])

    # single fill value
    f = funcs.StepFunction(x, x, fill_value=-100)
    np.testing.assert_allclose(f([-10, 30]), [-100, -100])

    # tuple fill value
    f = funcs.StepFunction(x, x, fill_value=(-100, -200))
    np.testing.assert_allclose(f([-10, 30]), [-100, -200])


def test_ecdf():
    obs = np.arange(20)

    # step function
    e = funcs.ECDF(obs)
    np.testing.assert_allclose(e([-1, 0, 0.5, 0.8, 1, 7.5, 18.8, 19, 19.5]),
                               [0, 1/20, 1/20, 1/20, 2/20, 8/20, 19/20,
                                1, 1])
    np.testing.assert_equal(e.observations, obs)

    # linear interpolated function
    e = funcs.ECDF(obs, interp=1)
    np.testing.assert_allclose(e([-1, 0, 0.5, 0.8, 1, 7.5, 18.8, 19, 19.5]),
                               [0, 1/20, 1.5/20, 1.8/20, 2/20, 8.5/20, 19.8/20,
                                1, 1])
