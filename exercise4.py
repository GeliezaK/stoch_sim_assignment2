import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DES import DES


def compute_avg_waiting_time(mu, n_customers, n_servers, rhov, dist):
    """Compute average waiting time per configuration (Number of servers, Rho, Service time distribution)"""
    n_simulations = 25
    lambd_c = rhov * n_servers * mu

    des = DES(mu, n_customers, n_servers, dist)
    data = np.zeros((n_simulations, n_customers))
    # Simulate n_simulations times and collect results
    for i in range(n_simulations):
        data[i, :] = des.experiment(lambd_c)
    return np.mean(data), np.var(data)


def simulate():
    """Simulate multiple times for different rhos/lambdas and numbers of servers. Return the average waiting times of
     each configuration."""

    rho_values = np.arange(0.05, 1, 0.05)
    distributions = ["markov", "deterministic", "hyper"]

    df = pd.DataFrame(columns=["n_servers", "service_rate_distribution", "rho", "avg_waiting_time", "var_waiting_time"])

    # Run n_simulations simulations for every configuration (n_servers, service_rate_distribution, rho)
    for servers_i, n_servers in enumerate(n_servers_values):
        print("Number of servers : ", n_servers)
        for rho_i, rhov in enumerate(rho_values):
            print("Rho = ", rhov)
            for dist_name in distributions:
                print("Service Rate Distribution =", dist_name)
                avg_waiting_time, variance = compute_avg_waiting_time(mu, n_customers, n_servers, rhov, dist_name)
                newrow = pd.DataFrame([[n_servers, dist_name, rhov, avg_waiting_time, variance]],
                                      columns=["n_servers", "service_rate_distribution", "rho", "avg_waiting_time",
                                               "var_waiting_time"])
                df = pd.concat([df, newrow], ignore_index=True)
    return df


def calculate_error(S, n):
    """Calculate the error of the 95%-confidence interval."""
    return 1.96 * (np.sqrt(S) / np.sqrt(n))


def plot_results(df):
    """Plot the mean waiting times against the number of servers and the rho values."""
    markov = df[df['service_rate_distribution'] == 'markov']
    deterministic = df[df['service_rate_distribution'] == 'deterministic']
    hyper = df[df['service_rate_distribution'] == 'hyper']

    plt.figure(figsize=(7, 8), layout="tight")
    # Markov
    plt.subplot(311)
    for i in n_servers_values:
        data = markov[markov['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="-", marker="o", label=f"{i}")
    plt.legend(title="Number of servers", loc="upper left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title(f"Exponential (mean = 1)")
    plt.ylabel(r"$\hat E(W)$")
    # Deterministic
    plt.subplot(312)
    for i in n_servers_values:
        data = deterministic[deterministic['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="-", marker="o", label=f"{i}")
    plt.legend(title="Number of servers", loc="upper left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel(r"$\hat E(W)$")
    plt.title(f"Deterministic (mean = 1)")
    # Hyper
    plt.subplot(313)
    for i in n_servers_values:
        data = hyper[hyper['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="-", marker="o", label=f"{i}")
    plt.legend(title="Number of servers", loc="upper left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel(r"$\hat E(W)$")
    plt.title("Hyperexponential (mean = 2)")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times_distributions_mu1_log.png")
    plt.show()


if __name__ == '__main__':
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    n_servers_values = [1, 2, 4]
    n_customers = 1000
    #plt.style.use('seaborn-v0_8-paper')

    # Run simulation with different rho/num_server - configurations
    # df = simulate()
    # df.to_csv("Simulated_data_nservers_rho_dist_mu1.csv")
    df = pd.read_csv("Simulated_data_nservers_rho_dist_mu1.csv")
    # Plot simulated results
    plot_results(df)
