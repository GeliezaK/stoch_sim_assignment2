import random
import pandas as pd
import simpy
import numpy as np
import matplotlib.pyplot as plt


def arrivals(env, n_costumers, lambd, mu, server, wait_times):
    for i in range(n_costumers):
        e = event(env, server, mu, wait_times)
        env.process(e)
        inter_arrival_time = random.expovariate(lambd)
        yield env.timeout(inter_arrival_time)


def event(env, server, mu, wait_times):
    arrive = env.now

    with server.request() as req:
        yield req
        
        service_duration = random.expovariate(mu)
        end_queue = env.now
        wait_times.append(end_queue - arrive)

        yield env.timeout(service_duration)

def experiment(l, mu, n_servers, n_customers):
    print("Rho = ", np.round(l / (mu*n_servers), 3))
    wait_times = []
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=n_servers)
    env.process(arrivals(env, n_customers, l, mu, server, wait_times))
    env.run()
    return wait_times


if __name__ == '__main__':
    lambd_values = np.arange(0.2, 1, 0.05)
    # [0.2, 0.6, 0.7, 0.8, 0.9]  # arrival rate
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    n_simulations = 25

    n_servers_values = [1, 2, 4]
    n_customers = 1000

    # df = pd.DataFrame()
    data = np.zeros((len(n_servers_values), len(lambd_values), n_simulations, n_customers))

    for servers_i, n_servers in enumerate(n_servers_values):
        for lambd_i, lambd in enumerate(lambd_values):
            for simulation in range(n_simulations):
                lambd_c = lambd*n_servers
                wait_times = experiment(lambd_c, mu, n_servers, n_customers)
                data[servers_i, lambd_i, simulation] = wait_times
                # df[f"{n_servers}"] = wait_times

    print(data)
    all_means = []
    for n_servers in data:
        all_means.append(np.mean(n_servers, axis=(1, 2)))
        # means = [np.mean(array, axis=(1, 2)) for array in n_servers]
        # all_means.append(means)

    for i, mean in enumerate(all_means):
        plt.plot(lambd_values, mean, "-o", label=n_servers_values[i])
    plt.legend(title="Number of servers")
    plt.title("Simulated mean waiting times")
    plt.ylabel("Average waiting times E(W)")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times.png")
    
    plt.show()
    # print(df)
    # print(df.mean())
