import pandas as pd
import numpy as np


def normal_dist(loc: float, scale: float, size: int, function) -> pd.DataFrame:
    """
    Generates random samples from normal distribution. 

    Args:
        loc (float): Mean
        scale (float): Standard deviation
        size (int): Sample size
        function (_type_): Function applied to each draw, e.g. def func(xi: float) -> list.

    Returns:
        pd.DataFrame: Transformed samples. 
    """
    x = np.random.normal(loc=loc, scale=scale, size=size)
    return pd.DataFrame([function(xi) for xi in x])


def uni_dist(low: float, high: float, size: int, function) -> pd.DataFrame:
    """
    Generates random samples from normal distribution. 

    Args:
        low (float): Lower limit
        high (float): Upper limit
        size (int): Sample size
        function (_type_): Function applied to each draw, e.g. def func(xi: float) -> list.

    Returns:
        pd.DataFrame: Transformed samples.
    """
    x = np.random.uniform(low=low, high=high, size=size)
    return pd.DataFrame([function(xi) for xi in x])