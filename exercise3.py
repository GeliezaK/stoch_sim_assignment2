import pandas as pd
from scipy.stats import ttest_ind
from DES import DES
from exercise4 import compute_avg_waiting_time, calculate_error
import matplotlib.pyplot as plt
import numpy as np

def simulate():
    """Simulate multiple times for different rhos/lambdas and numbers of servers. Return the average waiting times of
     each configuration."""

    rho_values = np.arange(0.05, 1, 0.05)
    df = pd.DataFrame(columns=["sorting", "rho", "avg_waiting_time", "var_waiting_time"])

    # Run n_simulations simulations for every configuration (n_servers, rho)
    for i, sorting_method in enumerate(sorting_methods):
        print("Sorting method: ", sorting_method)
        for rho_i, rhov in enumerate(rho_values):
            print("Rho = ", rhov)
            avg_waiting_time, var_waiting_time = compute_avg_waiting_time(mu, n_customers, n_servers, rhov, 'markov', sorting_method=sorting_method)
            newrow = pd.DataFrame([[sorting_method, rhov, avg_waiting_time, var_waiting_time]],
                                  columns=["sorting_method", "rho", "avg_waiting_time", "var_waiting_time"])
            df = pd.concat([df, newrow], ignore_index=True)
    return df


def plot_results(df):
    """Plot the mean waiting times against the number of servers and the rho values."""
    for i in sorting_methods:
        data = df[df['sorting_method'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, lolims=True, capsize=2,
                     linestyle="-", marker="o", label=f"{i}")
    plt.legend(title="Number of servers")
    plt.yscale('log')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title(f"Simulated Average Waiting Times \nfor Different Numbers of Servers")
    plt.ylabel(r"$\hat E(W)$")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times_exercise3.png")
    plt.show()


if __name__ == '__main__':
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    sorting_methods = ["none", "shortest_first"]
    n_customers = 1000
    n_servers = 1
    plt.style.use('seaborn-v0_8-paper')

    # statistical_analysis()
    df = simulate()
    df.to_csv("Simulated_data_sorting_rho_mu1.csv")
    # df = pd.read_csv("Simulated_data_nservers_rho_mu1.csv")
    print(df.head())
    plot_results(df)