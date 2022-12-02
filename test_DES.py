import numpy as np
import random
import matplotlib.pyplot as plt
from DES import hyperexp_sampler


def test_hyperexp_sampler():
    num_trials = 10000
    res_hyper = np.zeros(num_trials)
    for i in range(0, num_trials):
        # lambda = 1/mean service time
        res_hyper[i] = hyperexp_sampler([0.75, 1], [1, 0.2])

    # Expected means of distributions are 2, 1, 5
    assert np.isclose(np.mean(res_hyper), 2, rtol=0.1), f"Expected mean was 2, but actual mean is {np.mean(res_hyper)}"
