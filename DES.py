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
    """Run a single DES simulation with the parameters rho=l/mu, number of servers n_servers and number of customers
    n_customers. Return a list (of size n_customers) that stores the waiting times per customer. """
    wait_times = []
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=n_servers)
    env.process(arrivals(env, n_customers, l, mu, server, wait_times))
    env.run()
    return wait_times


def welchs_test(groupA, groupB):
    """Perform Welch's t-test since the variances are not equal accross groups. """
    return ttest_ind(groupA, groupB, equal_var=False)


def statistical_analysis():
    """Perform Statistical test with 2 vs. 1 server and 4 vs. 1 server"""
    df = pd.DataFrame()
    lambd = 0.9
    for c in n_servers_values:
        lambd_c = lambd * c
        wait_times = experiment(lambd_c, mu, c, n_customers)
        df[f"{c}"] = wait_times
    print("Var: \n", df.var())
    print("Mean: \n", df.mean())
    res2 = welchs_test(df["2"], df["1"])
    res4 = welchs_test(df["4"], df["1"])
    print("Welch test results 2 vs 1: ", res2.statistic, ", p-value: ", res2.pvalue)
    print("Welch test results 4 vs 1: ", res4.statistic, ", p-value: ", res4.pvalue)


def simulate():
    """Simulate multiple times for different rhos/lambdas and numbers of servers. Return the average waiting times of
     each configuration."""
    rho_values = np.arange(0.05, 1, 0.05) #effectively, this manipulates rho values since mu is always constant
    n_simulations = 25
    data = np.zeros((len(n_servers_values), len(rho_values), n_simulations, n_customers))
    for servers_i, n_servers in enumerate(n_servers_values):
        print("Number of servers : ", n_servers)
        for rho_i, rhov in enumerate(rho_values):
            lambd_c = rhov * n_servers * mu
            print("Rho = ", rhov)
            # Asser that rho=lambda/mu*n still holds
            assert rhov == lambd_c/(n_servers * mu)
            # Simulate n_simulations time and collect results
            for simulation in range(n_simulations):
                wait_times = experiment(lambd_c, mu, n_servers, n_customers)
                data[servers_i, rho_i, simulation] = wait_times

    # Calculate means over all simulations and over all customers in each simulation
    all_means = []
    for n_servers in data:
        all_means.append(np.mean(n_servers, axis=(1, 2)))

    return rho_values, all_means


def plot_results(rho_values, all_means):
    """Plot the mean waiting times against the number of servers and the rho values."""
    for i, mean in enumerate(all_means):
        plt.plot(rho_values, mean, "-o", label=n_servers_values[i]) #lambda values are equal to rho values since we keep mu=1
    plt.legend(title="Number of servers")
    plt.xticks(np.arange(0,1.1,0.1))
    plt.title("Simulated Average Waiting Times")
    plt.ylabel("Average waiting times E(W)")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times.png")
    plt.show()


if __name__ == '__main__':
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    n_servers_values = [1, 2, 4]
    n_customers = 1000

    # Welch test: 2 vs 1 server, 4 vs 1 server
    statistical_analysis()
    # Run simulation with different rho/num_server - configurations
    rho_values, all_means = simulate()
    # Plot simulated results
    plot_results(rho_values, all_means)

