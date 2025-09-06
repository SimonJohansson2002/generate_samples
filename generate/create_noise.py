import pandas as pd
import numpy as np

def get_noise(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """
    Generates white noise for the samples. 

    Args:
        df (pd.DataFrame): the data to which we add white noise
        scale (float): standard deviation for the white noise

    Returns:
        pd.DataFrame: white noise with the same dimensions as input df. 
    """
    n_rows, n_cols = df.shape
    columns = df.columns

    noise = np.random.normal(loc=0, scale=scale, size=(n_rows, n_cols))
    return pd.DataFrame(noise, columns=columns)


def add_noise(df: pd.DataFrame, scale: float, alpha: float) -> tuple[pd.DataFrame]:
    """
    Adds white noise to the given dataframe. 

    Args:
        df (pd.DataFrame): dataframe
        scale (float): standard deviation for the white noise
        alpha (float): diffusion rate

    Returns:
        tuple[pd.DataFrame]: new dataframe with added noise and the noise added
    """

    noise = get_noise(df, scale)
    df_noise = np.sqrt(alpha) * df + np.sqrt(1-alpha) * noise
    return df_noise, noise


def samples_with_noise(infile: str, scale: float, t: int, alpha_cumprod: np.cumprod, beta_start: float = 1e-4, beta_end: float = 0.02) -> pd.DataFrame:
    """
    Adds white noise to the real data. Uses linear diffusion schedule. 
    alpha = 1 - beta
    x1 = np.sqrt(alpha) * x0 + np.sqrt(1-alpha) * noise

    Args:
        infile (str): Path to real data
        scale (float): Standard deviation for white noise
        t (int): Maximum length of the stochastic process. Defaults to 1000.
        alpha_cumprod (np.cumprod): Compounded multiplication of aplha.
        beta_start (float, optional): Initial diffusion rate for white noise being added. Defaults to 1e-4.
        beta_end (float, optional): End diffusion rate for white noise being added. Defaults to 0.02.

    Returns (pd.DataFrame): samples with noise
    """
    df = pd.read_csv(infile)

    df_noise, noise = add_noise(df_noise, float(scale), alpha_cumprod[t])

    # Add noise columns
    for col in noise.columns:
        noise_col_name = f'Noise {col}'
        df_noise[noise_col_name] = noise[col]

    return df_noise

if __name__=='__main__':
    infile = 'real_samples/gaussian_test.csv'
    outfile = 'gaussian_noise_test.csv'
    scale = 1   # standard deviation for white noise
    
    df_noise = samples_with_noise(infile, outfile, scale)

    df_noise.to_csv(f'noise_samples/{outfile}', index=False)