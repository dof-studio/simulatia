## simulatia
## version 0.0.1
## 
## poisson process.py
## DOF Studio 2024
## Apache License Version 2.0

import numpy as np
from distributions import exponential

def poisson_process(lmbda, time = 1):
    '''
    Simulates a Poisson process with rate lmbda and time horizon T.

    Parameters:
        lmbda (float): The rate of the Poisson process.
        time (float): The time horizon of the simulation.

    Returns:
        arrival_times (numpy.ndarray): The arrival times of the Poisson process.
        intervals (numpy.ndarray): The intervals between arrivals.
    '''

    # Global intervals
    intervals = []

    while len(intervals) == 0 or np.sum(intervals) < time:
        
        # Generate exponential random numbers
        inter_arrival_times = exponential(1/lmbda, int(time/lmbda))

        # Append the intervals
        intervals.extend(inter_arrival_times)

    # Cumsum the intervals
    arrival_times = np.cumsum(inter_arrival_times)

    # Remove the arrivals that exceed the time
    arrival_times = arrival_times[arrival_times <= time]

    # Remove the relative intervals
    intervals = intervals[:len(arrival_times)]

    return np.array(arrival_times), np.array(intervals)


# Test
if __name__ == "__main__":
    
    # calculate the conditional distribution of the number of arrivals within 15 minutes
    numbaa = []
    for i in range(1000000):
        arrivals, intervals = poisson_process(1/6, 60)
        if len(arrivals) == 10:
            arrivals = arrivals[arrivals <= 15]
            numbaa.append(len(arrivals))
    print("mean=", np.mean(numbaa))
    print("std_err=", np.std(numbaa) / np.sqrt(len(numbaa)))
    print("95%CI=", np.mean(numbaa) - 1.96*np.std(numbaa) / np.sqrt(len(numbaa)), np.mean(numbaa) + 1.96*np.std(numbaa) / np.sqrt(len(numbaa)))
  
