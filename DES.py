import random
import pandas as pd
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def __markov_sampler__(lambd):
    """Return a value from a Markov distribution A(t) = 1 - e^(-lambd*t)."""
    return random.expovariate(lambd)


def __deterministic_sampler__(value):
    """Return a value from the deterministic distribution, i.e. the constant value."""
    return value


def __hyperexp_sampler__(distribution, lambdas):
    """Return a composed exponential distribution.

    :param distribution: array-like, contains the cumulative probabilities for each exponential distribution.
    :param lambdas: array-like, contains the lambdas for each exponential distribution."""

    assert len(distribution) == len(lambdas)
    # Assert that distrtibution contains cumulative values
    assert all(0 < p <= 1 for p in distribution)

    # Draw random number from (0,1)
    rand = random.random()
    ind = np.argmax(np.array(distribution) > rand)
    return random.expovariate(lambdas[ind])


class DES:
    def __init__(self, mu, n_customers, n_servers, service_rate_dist):
        self.mu = mu
        self.n_customers = n_customers
        self.n_servers = n_servers
        self.service_rate_dist = service_rate_dist

    def __arrivals__(self, env, lambd, server, wait_times):
        """
        :param env:
        :param n_costumers:
        :param lambd:
        :param server:
        :param wait_times:
        :param service_rate_dist: Function that returns a random variable from a certain distribution.
        :return:
        """
        for i in range(self.n_customers):
            e = self.__event__(env, server, wait_times)
            env.process(e)
            inter_arrival_time = random.expovariate(lambd)
            yield env.timeout(inter_arrival_time)

    def __event__(self, env, server, wait_times):
        """
        :param service_rate_dist: Function that generates a random number from the service rate distribution.
        :param env:
        :param server:
        :param wait_times:
        :return:
        """
        # Init service rate distribution
        dist = self.__init_service_rate_dist__(self.service_rate_dist)
        arrive = env.now

        with server.request() as req:
            yield req

            service_duration = dist()
            end_queue = env.now
            wait_times.append(end_queue - arrive)

            yield env.timeout(service_duration)

    def experiment(self, l):
        """Run a single DES simulation with the parameters rho=l/mu, number of servers n_servers and number of customers
        n_customers. Return a list (of size n_customers) that stores the waiting times per customer. """
        assert 0 <= l / (self.mu * self.n_servers) <= 1
        wait_times = []
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=self.n_servers)
        env.process(self.__arrivals__(env, l, server, wait_times))
        env.run()
        return wait_times

    def __init_service_rate_dist__(self, name):
        """Initialize the service rate distribution with mean 2.

        :param name: String, the name of the service name distribution. Either "markov", "deterministic" or "hyper"
        :return: a function that generates a random number from the desired distribution
        """
        if name == "markov":
            dist = lambda: __markov_sampler__(1 / self.mu)
        elif name == "hyper":
            dist = lambda: __hyperexp_sampler__([0.75, 1], [5, 0.2])
        else:
            dist = lambda: __deterministic_sampler__(self.mu)
        return dist


if __name__ == '__main__':
    des = DES(1, 1000, 2, 'hyper')
    wait_times = des.experiment(0.9)
    print(wait_times)
