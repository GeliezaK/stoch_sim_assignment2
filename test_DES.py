import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from DES import DES


def test_hyperexp_sampler():
    num_trials = 10000
    res_hyper = np.zeros(num_trials)
    for i in range(0, num_trials):
        # lambda = 1/mean service time
        res_hyper[i] = hyperexp_sampler([0.75, 1], [1, 0.2])

    # Expected means of distributions are 2, 1, 5
    assert np.isclose(np.mean(res_hyper), 2, rtol=0.1), f"Expected mean was 2, but actual mean is {np.mean(res_hyper)}"


def test_init_service_rate_dist():
    dist = init_service_rate_dist("markov", 0.9)
    assert dist() != dist()
    dist = init_service_rate_dist("hyper")
    assert dist() != dist()
    dist = init_service_rate_dist("deterministic", 5)
    assert dist() == dist() == 5

def test_pandas_concat():
    df = pd.DataFrame(columns=["n_servers", "service_rate_distribution", "rho", "avg_waiting_time"])
    newrow = pd.DataFrame([[2, 'markov', 0.5, 7]], columns=["n_servers", "service_rate_distribution", "rho", "avg_waiting_time"])
    df = pd.concat([df, newrow], ignore_index=True)
    print(df)