import datetime as dt

import numpy as np
from numpy.testing import assert_allclose

from orbit_predictor.predictors.numerical import J2Predictor


def test_benchmark_numeric(benchmark):
    sma = 6780
    ecc = 0.001
    inc = 28.5
    raan = 67.0
    argp = 355.0
    ta = 250.0

    epoch = dt.datetime(2000, 1, 1, 12, 0)

    predictor = J2Predictor(sma, ecc, inc, raan, argp, ta, epoch)

    expected_position = np.array([2085.9287615146, -6009.5713894563, -2357.3802307070])
    expected_velocity = np.array([6.4787522759177, 3.2366136616580, -2.5063420188165])

    when_utc = epoch + dt.timedelta(hours=3)

    position_eci, velocity_eci = benchmark(predictor._propagate_eci, when_utc=when_utc)

    assert_allclose(position_eci, expected_position, rtol=1e-2)
    assert_allclose(velocity_eci, expected_velocity, rtol=1e-2)
