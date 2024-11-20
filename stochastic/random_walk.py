## simulatia
## version 0.0.1
## 
## random walk.py
## DOF Studio 2024
## Apache License Version 2.0

import numpy as np
from distributions import bernoulli

def random_walk(p, time = 100):
    '''
    Simulates a random walk with probability p and time horizon T.

    Parameters:
        p (float): The probability of moving to the right.
        time (float): The time horizon of the simulation.

    Returns:
        positions (numpy.ndarray): The positions of the random walk.
        step_length (numpy.ndarray): The step_length (either 1 or -1) between steps.
    '''
    
    # Global increments
    increments = bernoulli(p, time)

    # Cumsum the increments
    realized_positions = np.cumsum(increments)

    return np.array(realized_positions), np.array(increments)

def unit_random_walk(p, time = 100):
    '''
    Simulates a unit random walk approxiamting W(1) with probability p and number of steps T.
    Note, it is simulating a W(t) where t = 1. But with different steps of reaching to W(1).

    Parameters:
        p (float): The probability of moving to the right.
        time (float): The number of steps of the simulation.

    Returns:
        positions (numpy.ndarray): The positions of the random walk.
        step_length (numpy.ndarray): The step_length (either 1 or -1) between steps.
    '''
    
    # Global increments
    increments = bernoulli(p, time)

    # Cumsum the increments
    positions = np.cumsum(increments)

    # Rescale the positions
    realized_positions = np.divide(positions, np.sqrt(time + 1))
    # Constant scalling, making variance to 1

    return np.array(realized_positions), np.array(increments)

def scaled_random_walk(p, time = 100):
    '''
    Simulates a scaled random walk with probability p and time horizon T.
    Note the simulated random walk has a CONSTANT variance of 1.

    Parameters:
        p (float): The probability of moving to the right.
        time (float): The time horizon of the simulation.

    Returns:
        positions (numpy.ndarray): The positions of the random walk.
        step_length (numpy.ndarray): The step_length (either 1 or -1) between steps.
    '''
    
    # Global increments
    increments = bernoulli(p, time)

    # Cumsum the increments
    positions = np.cumsum(increments)

    # Rescale the positions
    realized_positions = np.divide(positions, np.sqrt(np.arange(1, time + 1)))
    # Step-wise scaling, making the variance equal everywhere

    return np.array(realized_positions), np.array(increments)


# Test
if __name__ == "__main__":
    
    # calculate the conditional terminal expectation of the random walk if the max is greater than 50
    numbaa = []
    for i in range(1000000):
        realized, steps = random_walk(0.5, 100)
        if max(realized) >= 50:
            numbaa.append(realized[99])
    print("mean=", np.mean(numbaa))
    print("std_err=", np.std(numbaa) / np.sqrt(len(numbaa)))
    print("95%CI=", np.mean(numbaa) - 1.96*np.std(numbaa) / np.sqrt(len(numbaa)), np.mean(numbaa) + 1.96*np.std(numbaa) / np.sqrt(len(numbaa)))
  
