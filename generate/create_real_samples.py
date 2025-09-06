import pandas as pd
import numpy as np


def normal_dist(loc: float, scale: float, size: int, function) -> list[list[float]]:
    """
    Generates random samples from normal distribution. 

    Args:
        loc (float): Mean
        scale (float): Standard deviation
        size (int): Sample size
        function (_type_): Function applied to each draw.

    Returns:
        list[list[float]]: Transformed samples. 
    """
    x = np.random.normal(loc=loc, scale=scale, size=size)
    return [function(xi) for xi in x]


def uni_dist(low: float, high: float, size: int, function) -> list[list[float]]:
    """
    Generates random samples from normal distribution. 

    Args:
        low (float): Lower limit
        high (float): Upper limit
        size (int): Sample size
        function (_type_): Function applied to each draw.

    Returns:
        list[list[float]]: Transformed samples.
    """
    x = np.random.uniform(low=low, high=high, size=size)
    return [function(xi) for xi in x]


if __name__=='__main__':
    a = 0   # loc/lower (normal/uniform)
    b = 1   #scale/upper (normal/uniform)
    size = 1000
    file_name = 'gaussian_test'

    def func(xi: float) -> list:
        """
        Design your function for one sample. You can 

        Args:
            xi (float): the random value generated in one of the functions above.

        Returns:
            list: the structure of one sample, e.g. [xi, xi * 2]
        """
        
        return [xi, xi * 2]

    samples = pd.DataFrame(normal_dist(a, b, size, func))

    samples.to_csv(f'real_samples/{file_name}.csv', index=False)