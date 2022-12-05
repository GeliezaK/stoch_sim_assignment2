import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from DES import DES


def compute_avg_waiting_time(mu, n_customers, n_servers, rhov, dist):
    """Compute average waiting time per configuration (Number of servers, Rho, Service time distribution)"""
    n_simulations = 25
    lambd_c = rhov * n_servers * mu
    print("Lambda: ", lambd_c, ", Rho: ", np.round(lambd_c / (mu * n_servers), 3))

    des = DES(mu, n_customers, n_servers, dist)
    data = np.zeros((n_simulations, n_customers))
    # Simulate n_simulations times and collect results
    for i in range(n_simulations):
        data[i, :] = des.experiment(lambd_c)
    return np.mean(data), np.var(data)


def simulate():
    """Simulate multiple times for different rhos/lambdas and numbers of servers. Collect the average waiting
    times and the variance of each measurement in a dataframe.

    :returns df: Dataframe with a row for each configuration (rho, nservers, service rate distribution)"""

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
    """Calculate the error of the 95%-confidence interval.
    :param S: variance
    :param n: number of observations"""
    return 1.96 * (np.sqrt(S) / np.sqrt(n))


def plot_results(df):
    """Plot the mean waiting times against the number of servers and the rho values."""
    markov = df[df['service_rate_distribution'] == 'markov']
    deterministic = df[df['service_rate_distribution'] == 'deterministic']
    hyper = df[df['service_rate_distribution'] == 'hyper']

    plt.figure(figsize=(6, 10), layout="tight")
    # Markov
    plt.subplot(311)
    for i in n_servers_values:
        data = markov[markov['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="--", marker="o", markersize=3, label=f"{i}")
    plt.legend(title="Number of servers", loc="upper left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylim([0, 800])
    plt.title(f"Exponential")
    plt.ylabel(r"$\hat E(W)$")
    # Deterministic
    plt.subplot(312)
    for i in n_servers_values:
        data = deterministic[deterministic['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="--", marker="o", markersize=3, label=f"{i}")
    plt.legend(title="Number of servers", loc="upper left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel(r"$\hat E(W)$")
    plt.ylim([0, 800])
    plt.title(f"Deterministic")
    # Hyper
    plt.subplot(313)
    for i in n_servers_values:
        data = hyper[hyper['n_servers'] == i]
        errors = calculate_error(data['var_waiting_time'], n_customers)
        plt.errorbar(data['rho'], data['avg_waiting_time'], yerr=errors, capsize=2,
                     linestyle="--", marker="o", markersize=3, label=f"{i}")
    plt.legend(title="Number of servers", loc="upper left")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel(r"$\hat E(W)$")
    plt.ylim([0, 800])
    plt.title("Hyperexponential")
    plt.xlabel(r"System load ($\rho$)")
    plt.savefig("figures/Simulated_waiting_times_distributions_mu2.png")
    plt.show()


def welchs_test(groupA, groupB):
    """Perform Welch's t-test if the variances are not equal accross groups. """
    return ttest_ind(groupA, groupB, equal_var=False)


def get_waiting_times(rho):
    """Get the waiting times for each configuration (number of servers, service rate distribution). """
    df = pd.DataFrame()
    distributions = ['markov', 'deterministic', 'hyper']

    for dist in distributions:
        for c in n_servers_values:
            lambd_c = rho * c * mu
            des = DES(mu, n_customers, c, dist)
            wait_times = des.experiment(lambd_c)
            df[f"{dist}_{c}"] = wait_times
    return df


def print_stats(df, dist1, dist2, c):
    """Perform a Welch-t-test between dist1 and dist2 (with c servers) and print the results."""
    res = welchs_test(df[f"{dist1}_{c}"], df[f"{dist2}_{c}"])
    print(f"Mean of {dist1}: ", df[f"{dist1}_{c}"].mean())
    print(f"Mean of {dist2}: ", df[f"{dist2}_{c}"].mean())
    print(f"Std of {dist1}: ", df[f"{dist1}_{c}"].std())
    print(f"Std of {dist2}: ", df[f"{dist2}_{c}"].std())
    print(f"Welch test results {dist1} vs {dist2}, c={c}: ", res.statistic, ", p-value: ", res.pvalue)


def plot_boxplots(df):
    """
     Plot grouped boxplots to show the distribution of waiting times (grouped by: number of servers,
     service rate distribution)
     Code from https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/"""
    ticks = ["1", "2", "4"]
    markov_data = [df['markov_1'], df['markov_2'], df['markov_4']]
    hyper_data = [df['hyper_1'], df['hyper_2'], df['hyper_4']]
    deterministic_data = [df['deterministic_1'], df['deterministic_2'], df['deterministic_4']]
    markov_plot = plt.boxplot(markov_data,
                                   positions=np.array(np.arange(len(markov_data))) * 3.0 - 0.6,
                                   widths=0.5)
    hyper_plot = plt.boxplot(hyper_data,
                                   positions=np.array(
                                       np.arange(len(hyper_data))) * 3.0,
                                   widths=0.5)
    det_plot = plt.boxplot(deterministic_data,
                             positions=np.array(
                                 np.arange(len(deterministic_data))) * 3.0 + 0.6,
                             widths=0.5)
    define_box_properties(markov_plot, '#D7191C', 'markov')
    define_box_properties(hyper_plot, '#2C7BB6', 'hyper')
    define_box_properties(det_plot, 'green', 'deterministic')

    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks)

    # set the limit for x axis
    plt.xlim(-3, len(ticks) * 3)
    plt.title(r"Waiting times $(\rho = 0.9)$"
              f"\nfor Different Service Rate Distributions")
    plt.xlabel("Number of servers")
    plt.ylabel("Waiting times")
    plt.savefig("Boxplots.png")
    plt.show()


def define_box_properties(plot_name, color_code, label):
    """Add color and label to the grouped boxplots."""
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()


if __name__ == '__main__':
    mu = 2  # server capacity (rate, e.g. 1.2 customers/unit time)
    n_servers_values = [1, 2, 4]
    n_customers = 1000

    # Statistical analysis
    df = get_waiting_times(0.9)
    # Store the results in csv to avoid multiple simulation
    df.to_csv("waiting_times_rho09.csv")
    # Read the results from csv to pandas if it has been simulated before
    #df = pd.read_csv("waiting_times_rho09.csv")
    print_stats(df, 'hyper', 'deterministic',4 )
    print_stats(df, 'hyper', 'deterministic',2 )
    print_stats(df, 'hyper', 'deterministic',1 )
    print_stats(df, 'hyper', 'markov',4 )
    print_stats(df, 'hyper', 'markov',2 )
    print_stats(df, 'hyper', 'markov',1 )
    print_stats(df, 'markov', 'deterministic',4 )
    print_stats(df, 'markov', 'deterministic',2 )
    print_stats(df, 'markov', 'deterministic',1 )
    plot_boxplots(df)


    # Run simulation with different rho/num_server - configurations
    df = simulate()
    df.to_csv("Simulated_data_nservers_rho_dist_mu2.csv")
    # df = pd.read_csv("Simulated_data_nservers_rho_dist_mu2.csv")
    # Plot simulated results
    plot_results(df)
