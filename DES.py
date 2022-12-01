import random

import pandas as pd
import simpy
import numpy as np


def arrivals(env, n_costumers, lambd, mu, server, wait_times):
    for i in range(n_costumers):
        e = event(env, 'Event_%02d' % i, server, mu, wait_times)
        env.process(e)
        inter_arrival_time = random.expovariate(lambd)
        yield env.timeout(inter_arrival_time)


def event(env, name, server, mu, wait_times):
    arrive = env.now

    with server.request() as req:
        yield req
        
        service_duration = random.expovariate(mu)
        end_queue = env.now
        wait_times.append(end_queue - arrive)

        yield env.timeout(service_duration)

def experiment(l, mu, n_servers, n_costumers):
    print("Rho = ", np.round(l / mu*n_servers, 3))
    wait_times = []
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=n_servers)
    env.process(arrivals(env, n_costumers, l, mu, server, wait_times))
    env.run()
    return wait_times

if __name__ == '__main__':
    lambd = 1  # arrival rate
    mu = 1.1  # server capacity/rate

    n_servers = [1, 2, 4]
    n_costumers = 100

    df = pd.DataFrame()

    for ind, c in enumerate(n_servers):
        lambd_c = lambd/c
        wait_times = experiment(lambd_c, mu, c, n_costumers)
        df[f"{c}"] = wait_times

    print(df)
    print(df.mean())


