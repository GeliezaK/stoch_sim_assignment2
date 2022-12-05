# DES simulation assignment 
In order to run the code of a file, run the __main__ function in the respective file. 

## plot_exp_wait_times.py
Calling the __main__ function will plot the calculated average waiting times for different rho and 1,2, and 4 servers (Figure 1 in the assignment). 

## DES class 
This class implements the discrete event simulation functionality. A DES object is initialized with the server capacity mu, the number of arrivals ("customers") of the whole simulation, the number of servers and the name of the service rate distribution. The only public method of the class is "experiment()" which can be given a parameter lambda. This method launches the SimPy-discrete event simulation of the queuing system that was initialized before. An example of how to initialize the DES and start a simulation is given in the file's __main__ function. It also plots the distribution and the mean of the waiting times returned from that simulation. 

## Exercise4.py
The code sets up simulations for different system configurations (rho, number of servers, service rate distribution). The function simulate() generates a pandas.Dataframe that contains the relevant simulation outputs (average waiting time and variance of waiting times) for each configuration (each row is a different configuration). The dataframe can be handed to the function plot_results() which will create Figure 3 in the assignment. The function get_waiting_times() returns another pandas.Dataframe in which each column stores all waiting times for a single configuration. The function plot_boxplots() will create the boxplots in Figure 4 in the assignment. The function print_stats() performs a Welch-test between the given groups and prints the results. 

## Exercise2.py
This code is similar to Exercise4.py but it performs the DES without comparing the service time distributions. The function statistical_analysis() performs Welch-t-tests of the average waiting times between 1,2 and 4 servers in M/M/c queues and prints the results. The function simulate() generates a pandas.Dataframe that contains the relevant simulation outputs (average waiting time and variance of waiting times) for each configuration (rho, numbers of servers). The dataframe can be handed to the function plot_results() which will create Figure 2 in the assignment.
