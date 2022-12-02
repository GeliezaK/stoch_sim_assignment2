import pandas as pd
from scipy.stats import ttest_ind
from DES import DES


def welchs_test(groupA, groupB):
    """Perform Welch's t-test since the variances are not equal accross groups. """
    return ttest_ind(groupA, groupB, equal_var=False)


def statistical_analysis():
    """Perform Statistical test with 2 vs. 1 server and 4 vs. 1 server"""
    df = pd.DataFrame()
    rho = 0.9

    for c in n_servers_values:
        lambd_c = rho * c * mu
        des = DES(mu, n_customers, c, 'markov')
        wait_times = des.experiment(lambd_c)
        df[f"{c}"] = wait_times

    res2 = welchs_test(df["2"], df["1"])
    res4 = welchs_test(df["4"], df["1"])
    print("Welch test results 2 vs 1: ", res2.statistic, ", p-value: ", res2.pvalue)
    print("Welch test results 4 vs 1: ", res4.statistic, ", p-value: ", res4.pvalue)

if __name__ == '__main__':
    mu = 2  # server capacity (rate, e.g. 1.2 customers/unit time)
    n_servers_values = [1, 2, 4]
    n_customers = 1000

    statistical_analysis()