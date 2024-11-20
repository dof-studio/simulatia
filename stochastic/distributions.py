## simulatia
## version 0.0.1
## 
## distribution.py
## DOF Studio 2024
## Apache License Version 2.0

import numpy as np

def bernoulli(p, n):
    '''
    Generate n Bernoulli random numbers with probability p.

    Parameters:
        p (float): The probability of success.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n Bernoulli random numbers.
    '''
    return np.random.binomial(1, p, n)

def binomial(k, p, n):
    '''
    Generate n binomial random numbers with parameters k and p.

    Parameters:
        k (int): The number of trials.
        p (float): The probability of success.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n binomial random numbers.
    '''
    return np.random.binomial(k, p, n)

def geometric(p, n):
    '''
    Generate n geometric random numbers with probability p.

    Parameters:
        p (float): The probability of success.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n geometric random numbers.
    '''
    return np.random.geometric(p, n)

def hypergeometric(N, K, n):
    '''
    Generate n hypergeometric random numbers with parameters N, K, and n.

    Parameters:
        N (int): The total number of items in the population.
        K (int): The number of items in the population with a specific characteristic.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n hypergeometric random numbers.
    '''
    return np.random.hypergeometric(N, K, n)

def uniform(a, b, n):
    '''
    Generate n uniform random numbers between a and b.

    Parameters:
        a (float): The lower bound of the uniform distribution.
        b (float): The upper bound of the uniform distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n uniform random numbers.
    '''
    return np.random.uniform(a, b, n)

def poisson(lmbda, n):
    '''
    Generate n Poisson random numbers with rate lmbda.

    Parameters:
        lmbda (float): The rate of the Poisson distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n Poisson random numbers.
    '''
    return np.random.poisson(lmbda, n)

def exponential(mu, n):
    '''
    Generate n exponential random numbers with mean mu.

    Parameters:
        mu (float): The mean of the exponential distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n exponential random numbers.
    '''
    return np.random.exponential(mu, n)

def gamma(alpha, beta, n):
    '''
    Generate n gamma random numbers with shape alpha and scale beta.

    Parameters:
        alpha (float): The shape parameter of the gamma distribution.
        beta (float): The scale parameter of the gamma distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n gamma random numbers.
    '''
    return np.random.gamma(alpha, beta, n)

def beta(alpha, beta, n):
    '''
    Generate n beta-distributed random numbers with shape parameters alpha and beta.

    Parameters:
        alpha (float): The first shape parameter of the beta distribution.
        beta (float): The second shape parameter of the beta distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n beta-distributed random numbers.
    '''
    return np.random.beta(alpha, beta, n)

def normal(mu, sigma, n):
    '''
    Generate n normal random numbers with mean mu and standard deviation sigma.

    Parameters:
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n normal random numbers.
    '''
    return np.random.normal(mu, sigma, n)

def lognormal(mu, sigma, n):
    '''
    Generate n log-normal random numbers with mean mu and standard deviation sigma.

    Parameters:
        mu (float): The mean of the log-normal distribution.
        sigma (float): The standard deviation of the log-normal distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n log-normal random numbers.
    '''
    return np.random.lognormal(mu, sigma, n)

def chi2(df, n):
    '''
    Generate n chi-squared random numbers with degrees of freedom df.

    Parameters:
        df (float): The degrees of freedom of the chi-squared distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n chi-squared random numbers.
    '''
    return np.random.chisquare(df, n)

def t(df, n):
    '''
    Generate n t-distributed random numbers with degrees of freedom df.

    Parameters:
        df (float): The degrees of freedom of the t-distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n t-distributed random numbers.
    '''
    return np.random.standard_t(df, n)

def f(df1, df2, n):
    '''
    Generate n F-distributed random numbers with degrees of freedom df1 and df2.

    Parameters:
        df1 (float): The numerator degrees of freedom of the F-distribution.
        df2 (float): The denominator degrees of freedom of the F-distribution.
        n (int): The number of random numbers to generate.

    Returns:
        numpy.ndarray: An array of n F-distributed random numbers.
    '''
    return np.random.f(df1, df2, n)
