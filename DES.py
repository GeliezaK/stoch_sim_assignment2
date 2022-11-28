import random
import simpy
import numpy as np

lambd = 1 #arrival rate
mu = 1.2 #server capacity/rate

n_servers = 2
n_costumers = 100

wait_times = []

def source(env, n_costumers, lambd, mu, server, wait_times):
    for i in range(n_costumers):
        p = process(env, 'Process_%02d' % i, server, mu, wait_times)
        env.process(p)
        wait = random.expovariate(lambd)
        yield env.timeout(wait)


def process(env, name, server, mu, wait_times):
    arrive = env.now
    print('%7.4f %s: Here I am' % (arrive, name))

    with server.request() as req:
        yield req
        
        process_duration = random.expovariate(mu)
        end_queue = env.now
        wait_times.append(end_queue - arrive)

        print('%s starting at %7.4f' % (name, end_queue))
        yield env.timeout(process_duration)
        print('%s finished at %7.4f' % (name, env.now))


env = simpy.Environment()
server = simpy.Resource(env, capacity=n_servers)
env.process(source(env, n_costumers, lambd, mu, server, wait_times))
env.run()

print(f"\nAverage wait time of {np.mean(wait_times)}")
