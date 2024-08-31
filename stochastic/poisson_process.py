import numpy as np

## 1. Generate exponential random numbers
def expnum(mu, n):
    return np.random.exponential(mu, n)

## 2. Generate Poisson random numbers
def poisson(lmbda, n):
    return np.random.poisson(lmbda, n)

## 3. Generate uniform random numbers
def uniform(a, b, n):
    return np.random.uniform(a, b, n)

## Simuate Poisson Process
def pois_process(lmbda, time = 1):
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
        inter_arrival_times = expnum(1/lmbda, int(time/lmbda))

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
    
    # calculate the cond
    numba = []
    for i in range(1000000):
        arr, intv = pois_process(1/6, 60)
        if len(arr) == 10:
            arr = arr[arr <= 15]
            numba.append(len(arr))
    print("mean=", np.mean(numba))
    print("std_err=", np.std(numba) / np.sqrt(len(numba)))
    print("95%CI=", np.mean(numba) - 1.96*np.std(numba) / np.sqrt(len(numba)), np.mean(numba) + 1.96*np.std(numba) / np.sqrt(len(numba)))
  
