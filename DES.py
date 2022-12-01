import random
import pandas as pd
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind



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
    wait_times = []
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=n_servers)
    env.process(arrivals(env, n_customers, l, mu, server, wait_times))
    env.run()
    return wait_times


def welchs_test(groupA, groupB):
    """Perform Welch's t-test since the variances are not equal accross groups. """
    return ttest_ind(groupA, groupB, equal_var=False)


if __name__ == '__main__':
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    
    n_servers_values = [1, 2, 4]
    n_customers = 1000

    # Perform Statistical test with 2 vs. 1 server and 4 vs. 1 server
    df = pd.DataFrame()
    lambd = 0.9
    
    for c in n_servers_values:
        lambd_c = lambd*c
        wait_times = experiment(lambd_c, mu, c, n_customers)
        df[f"{c}"] = wait_times

    print("Var: \n", df.var())
    print("Mean: \n", df.mean())

    res2 = welchs_test(df["2"], df["1"])
    res4 = welchs_test(df["4"], df["1"])
    print("Welch test results 2 vs 1: ", res2.statistic, ", p-value: ", res2.pvalue)
    print("Welch test results 4 vs 1: ", res4.statistic, ", p-value: ", res4.pvalue)
    
    
    # Plot Simulations for different lambdas 
    lambd_values = np.arange(0.2, 1, 0.05)
    n_simulations = 25
    data = np.zeros((len(n_servers_values), len(lambd_values), n_simulations, n_customers))

    for servers_i, n_servers in enumerate(n_servers_values):
        print("Number of servers : ", n_servers)
        for lambd_i, lambd in enumerate(lambd_values):
            lambd_c = lambd * n_servers
            print("Rho = ", np.round(lambd_c / (mu * n_servers), 3))
            for simulation in range(n_simulations):
                wait_times = experiment(lambd_c, mu, n_servers, n_customers)
                data[servers_i, lambd_i, simulation] = wait_times
               
    all_means = []
    for n_servers in data:
        all_means.append(np.mean(n_servers, axis=(1, 2)))
        
    for i, mean in enumerate(all_means):
        plt.plot(lambd_values, mean, "-o", label=n_servers_values[i])
    plt.legend(title="Number of servers")
    plt.title("Simulated mean waiting times")
    plt.ylabel("Average waiting times E(W)")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times.png")
    
    plt.show()

