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
    df = pd.DataFrame(columns=["n_servers", "rho", "avg_waiting_time", "var_waiting_time"])

    # Run n_simulations simulations for every configuration (n_servers, rho)
    for servers_i, n_servers in enumerate(n_servers_values):
        print("Number of servers : ", n_servers)
        for rho_i, rhov in enumerate(rho_values):
            print("Rho = ", rhov)
            avg_waiting_time, var_waiting_time = compute_avg_waiting_time(mu, n_customers, n_servers, rhov, 'markov')
            newrow = pd.DataFrame([[n_servers, rhov, avg_waiting_time, var_waiting_time]],
                                  columns=["n_servers", "rho", "avg_waiting_time", "var_waiting_time"])
            df = pd.concat([df, newrow], ignore_index=True)
    return df


def welchs_test(groupA, groupB):
    """Perform Welch's t-test since the variances are not equal accross groups. """
    return ttest_ind(groupA, groupB, equal_var=False)


def statistical_analysis():
    """Perform Statistical test with 2 vs. 1 server and 4 vs. 1 server"""
    df = pd.DataFrame()
    rho = 0.1

    for c in n_servers_values:
        lambd_c = rho * c * mu
        des = DES(mu, n_customers, c, 'markov')
        wait_times = des.experiment(lambd_c)
        df[f"{c}"] = wait_times

    res2 = welchs_test(df["2"], df["1"])
    res4 = welchs_test(df["4"], df["1"])
    print("Welch test results 2 vs 1: ", res2.statistic, ", p-value: ", res2.pvalue)
    print("Welch test results 4 vs 1: ", res4.statistic, ", p-value: ", res4.pvalue)


def plot_results(df):
    """Plot the mean waiting times against the number of servers and the rho values."""
    for i in n_servers_values:
        data = df[df['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="-", marker="o", label=f"{i}")
    plt.legend(title="Number of servers")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title(f"Simulated Average Waiting Times \nfor Different Numbers of Servers")
    plt.ylabel(r"$\hat E(W)$")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times_exercise2.png")
    plt.show()


if __name__ == '__main__':
    mu = 1  # server capacity (rate, e.g. 1.2 customers/unit time)
    n_servers_values = [1, 2, 4]
    n_customers = 100
    #plt.style.use('seaborn-v0_8-paper')

    statistical_analysis()
    #df = simulate()
    #df.to_csv("Simulated_data_nservers_rho_mu1.csv")
    df = pd.read_csv("Simulated_data_nservers_rho_mu1.csv")
    #print(df.head())
    plot_results(df)
