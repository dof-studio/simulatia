## simulatia
## version 0.0.1
##
## visualizers.py
## DOF Studio 2024
## Apache License Version 2.0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_process(
        simulation_func, 
        params, 
        time_horizon=1, 
        num_realizations=10, 
        process_name="",
        *,
        marker='.'):
    '''
    Visualizes a stochastic process by plotting multiple realizations and interval histograms.

    Parameters:
        simulation_func (callable): The stochastic process simulation function.
        params (dict): Parameters to pass to the simulation function.
        time_horizon (float): The time horizon for each realization.
        num_realizations (int): Number of realizations to simulate and plot.

    Returns:
        None
    '''
    plt.figure(figsize=(14, 6))

    # Plot multiple realizations
    plt.subplot(1, 2, 1)
    for i in range(num_realizations):
        realized_values, _ = simulation_func(time=time_horizon, **params)
        values = np.arange(1, len(realized_values) + 1)  # Value increases with each steps
        plt.plot(values, realized_values, marker=marker, label=f'Realization {i+1}' if i < 4 else None)
    plt.xlabel('Time')
    plt.ylabel('Realization')
    plt.title('Stochastic Process ' + process_name + ' Realizations')
    plt.grid(True)

    # Plot histogram of 2nd return value
    plt.subplot(1, 2, 2)
    al_second_values = []
    for _ in range(num_realizations):
        _, second_values = simulation_func(time=time_horizon, **params)
        al_second_values.extend(second_values)
    plt.hist(al_second_values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('2nd Return Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of 2nd Return Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return None

def animate_process_evolution(
        simulation_func, 
        params, 
        time_horizon=[1], 
        num_realizations=10, 
        process_name="",
        *,
        marker='.',
        interval=100,
        repeat_delay=1000):
    '''
    Animates the evolution of a stochastic process over a list of n values by displaying
    multiple realizations and updating the histogram of the second return values.

    Parameters:
        simulation_func (callable): The stochastic process simulation function.
        params (dict): Parameters to pass to the simulation function.
        time_horizon (list of int): A list of n values representing different steps or time points.
        num_realizations (int): Number of realizations to simulate and plot.
        process_name (str): Name of the stochastic process for titles.

    Returns:
        None
    '''
    # Precompute all realizations and second return values for efficiency
    realizations = []
    second_return_values = []
    for _ in range(num_realizations):
        realized_values, second_values = simulation_func(time=max(time_horizon), **params)
        realizations.append(realized_values)
        second_return_values.append(second_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_real = axes[0]
    ax_hist = axes[1]

    ax_real.set_xlabel('Time')
    ax_real.set_ylabel('Realization')
    ax_real.set_title(f'Stochastic Process {process_name} Realizations')
    ax_real.grid(True)
    lines = []
    for i in range(num_realizations):
        line, = ax_real.plot([], [], marker=marker, label=f'Realization {i+1}' if i < 4 else None)
        lines.append(line)
    if num_realizations <= 4:
        ax_real.legend()

    ax_hist.set_xlabel('2nd Return Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Histogram of 2nd Return Value')
    ax_hist.grid(True)
    hist_bins = 30
    hist_values = []

    plt.tight_layout()

    def init():
        for line in lines:
            line.set_data([], [])
        ax_hist.cla()
        ax_hist.set_xlabel('2nd Return Value')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of 2nd Return Value')
        ax_hist.grid(True)
        return lines

    def update(frame):
        n = time_horizon[frame]
        ax_real.set_xlim(1, max(time_horizon))
        ax_real.set_ylim(
            min([min(realization[:n]) for realization in realizations]) - 1,
            max([max(realization[:n]) for realization in realizations]) + 1
        )
        for i, line in enumerate(lines):
            x = np.arange(1, n + 1)
            y = realizations[i][:n]
            line.set_data(x, y)
        ax_hist.cla()
        current_second_values = [realizations[i][n-1] for i in range(num_realizations)]
        ax_hist.hist(current_second_values, bins=hist_bins, alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel('2nd Return Value')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of 2nd Return Value')
        ax_hist.grid(True)
        return lines

    anim = FuncAnimation(fig, update, frames=len(time_horizon), init_func=init,
                         blit=False, repeat=False, interval=interval, repeat_delay=repeat_delay)

    plt.show()

    return None
