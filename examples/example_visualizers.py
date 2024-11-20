## simulatia
## version 0.0.1
## 
## example_visualizers.py
## DOF Studio 2024
## Apache License Version 2.0
import numpy as np

# From simulatia, see: https://github.com/dof-studio/simulatia
from distributions import uniform
from visualizers import visualize_process, animate_process_evolution

# General relu to transform uniform numbers to directions
def grelu(x, thres=0.5, u_value=1, d_value=-1) -> np.ndarray:
    return np.where(x > thres, u_value, d_value)

# Simulation of a symmetric random walk
def symmetric_random_walk(p = 0.5, time = 100):
    
    # Generated uniform numbers
    unif = uniform(0, 1, n=time)

    # Get the direction by `grelu`
    directions = grelu(unif, p, 1, -1)

    # Accumulate them by cumsum
    positions = np.cumsum(directions, axis=0)

    return np.array(positions), np.array(directions)

# Simulation of a step-wise scaled symmetric random walk
def step_scaled_symmetric_random_walk(p = 0.5, time = 100):
     
    # Generated uniform numbers
    unif = uniform(0, 1, n=time)

    # Get the direction by `grelu`
    directions = grelu(unif, p, 1, -1)

    # Accumulate them by cumsum and scale
    positions = np.divide(np.cumsum(directions, axis=0), np.sqrt(np.arange(1, time + 1)))

    return np.array(positions), np.array(directions)

# Simulation of an approximation of a brownian motion
def unit_symmetric_random_walk(p = 0.5, time = 100):
     
    # Generated uniform numbers
    unif = uniform(0, 1, n=time)

    # Get the direction by `grelu`
    directions = grelu(unif, p, 1, -1)

    # Accumulate them by cumsum and scale
    positions = np.divide(np.cumsum(directions, axis=0), np.sqrt(time))

    return np.array(positions), np.array(directions)


# Entrypoint
if __name__ == "__main__":
    # Task I
    pos_10, dir_10 = symmetric_random_walk(time = 10)
    pos_100, dir_100 = symmetric_random_walk(time = 100)
    pos_1000, dir_1000 = symmetric_random_walk(time = 1000)
    pos_10000, dir_10000 = symmetric_random_walk(time = 10000)

    # Task II
    params = {"p":0.5}
    visualize_process(symmetric_random_walk, params, time_horizon=10, num_realizations=15, process_name="Symmetric Random Walk")
    visualize_process(symmetric_random_walk, params, time_horizon=100, num_realizations=15, process_name="Symmetric Random Walk")
    visualize_process(symmetric_random_walk, params, time_horizon=1000, num_realizations=15, process_name="Symmetric Random Walk")
    visualize_process(symmetric_random_walk, params, time_horizon=10000, num_realizations=15, process_name="Symmetric Random Walk")
    
    # Task III
    visualize_process(step_scaled_symmetric_random_walk, params, time_horizon=10000, num_realizations=15, process_name="Scaled Symmetric Random Walk", marker="")

    # Task More
    ns = list(range(100, 10100, 100))
    animate_process_evolution(unit_symmetric_random_walk, params, time_horizon=ns, num_realizations=100, process_name="Brownian Motion W(1)", interval=10, marker="")
    
