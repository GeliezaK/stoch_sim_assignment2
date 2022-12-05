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
    df = pd.DataFrame(columns=["sorting", "rho", "n_servers", "avg_waiting_time", "var_waiting_time"])

    # Run n_simulations simulations for every configuration (n_servers, rho)
    for i, sorting_method in enumerate(sorting_methods):
        print("Sorting method: ", sorting_method)
        for rho_i, rhov in enumerate(rho_values):
            for c in n_server_values:
                print("Rho = ", rhov)
                avg_waiting_time, var_waiting_time = compute_avg_waiting_time(mu, n_customers, c, rhov, 'markov', sorting_method=sorting_method)
                newrow = pd.DataFrame([[sorting_method, rhov, c, avg_waiting_time, var_waiting_time]],
                                  columns=["sorting_method", "rho", "n_servers", "avg_waiting_time", "var_waiting_time"])
                df = pd.concat([df, newrow], ignore_index=True)
    return df


def get_style(sorting_method, n_servers):
    if sorting_method=="none":
        marker = 'o'
        linestyle ='--'
    else:
        marker = 'x'
        linestyle= ':'
    # Get color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if n_servers == 1:
        color = colors[0]
    elif n_servers ==2:
        color = colors[1]
    else:
        color = colors[2]

    return marker, color, linestyle




def plot_results(df):
    """Plot the mean waiting times against the number of servers and the rho values."""
    plt.figure(figsize=(10,6))
    for ind, i in enumerate(sorting_methods):
        ax = plt.subplot(1,2,ind+1)
        for c in n_server_values:
            data = df[(df['sorting_method'] == i) & (df['n_servers'] == c)]
            marker, color, linestyle = get_style(i, c)
            errors = calculate_error(data['var_waiting_time'], n_customers)
            ax.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle=linestyle, marker=marker, color=color, label=f"{c}")
        plt.ylabel(r"$\hat E(W)$")
        ax.legend(title="Number of servers")
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.ylim(-0.5,14.5)
        if i == 'none':
            plt.title("FIFO")
        else:
            plt.title("SJF")
        plt.xlabel(r"System load ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times_exercise3.png")
    plt.show()


if __name__ == '__main__':
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    sorting_methods = ["none", "shortest_first"]
    n_customers = 1000
    n_server_values = [1, 2, 4]

    # statistical_analysis()
    #df = simulate()
    #df.to_csv("Simulated_data_sorting_rho_mu1.csv")
    df = pd.read_csv("Simulated_data_sorting_rho_mu1.csv")
    plot_results(df)