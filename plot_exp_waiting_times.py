import  matplotlib.pyplot as plt
import numpy as np

def delay_probability(c, p):
    """Equation 5.1 from Queuing Theory Background Material 2 (page 44) """
    factor1 = ((c * p)**c )/ np.math.factorial(c)
    series = [((c * p)**n)/np.math.factorial(n) for n in range(0, c)]
    factor21 = ((1-p) * sum(series))
    factor2 = 1/(factor21 + factor1)
    return factor1 * factor2


def expected_waiting_time(c, p, mu):
    """Equation 5.3 from Queuing Theory Background Material 2 (page 44)"""
    return delay_probability(c, p) * 1/ (1 - p) * 1/(c * mu)


def plot_expected_wait_times():
    """Calculate the average waiting times from the equations for different rho and plot them. """
    occupation_rates = np.arange(0.05,1,0.05)
    print(occupation_rates)
    plt.plot(occupation_rates, expected_waiting_time(1, occupation_rates, 1), 'o-', label="1")
    plt.plot(occupation_rates, expected_waiting_time(2, occupation_rates, 1), 'o-', label="2")
    plt.plot(occupation_rates, expected_waiting_time(4, occupation_rates, 1), 'o-', label="4")
    plt.legend(title="Number of Servers", loc="upper left")
    plt.ylabel("Average Waiting Times $E(W)$")
    plt.xticks(np.arange(0,1.1,0.1))
    plt.title("Calculated Average Waiting Times")
    plt.xlabel(r"Occupation rates of a single server ($\rho$)")
    plt.savefig("figures/calculated_waiting_times.png")
    plt.show()


if __name__ == '__main__':
    plot_expected_wait_times()