import pandas as pd
import numpy as np

def get_noise(df: pd.DataFrame, loc: float, scale: float) -> pd.DataFrame:
    """
    Generates white noise for the samples. 

    Args:
        df (pd.DataFrame): The data to which we add white noise
        loc (float): Mean for white noise
        scale (float): Ttandard deviation for the white noise

    Returns:
        pd.DataFrame: White noise with the same dimensions as input df. 
    """
    n_rows, n_cols = df.shape
    columns = df.columns

    noise = np.random.normal(loc=loc, scale=scale, size=(n_rows, n_cols))

    return pd.DataFrame(noise, columns=columns)


def add_noise(df: pd.DataFrame, loc: float, scale: float, t: np.array, alpha_cumprod: np.cumprod) -> tuple[pd.DataFrame]:
    """
    Adds white noise to the given dataframe. 

    Args:
        df (pd.DataFrame): Dataframe
        loc (float): Mean for white noise
        scale (float): Standard deviation for the white noise
        t (np.array): Matrix with timesteps for each sample
        alpha_cumprod (np.cumprod): Compounded diffusion rates

    Returns:
        tuple[pd.DataFrame]: New dataframe with added noise and the noise added
    """
    noise = get_noise(df, loc, scale)
    weights = np.sqrt(alpha_cumprod[t])[:, None]   # (n_samples, 1)
    df_noise = weights * df + np.sqrt(1 - alpha_cumprod[t])[:, None] * noise

    return df_noise, noise


def samples_with_noise(df: pd.DataFrame, loc: float, scale: float, t: np.array, alpha_cumprod: np.cumprod) -> pd.DataFrame:
    """
    Adds white noise to the real data. Uses linear diffusion schedule. 
    alpha = 1 - beta
    x1 = np.sqrt(alpha) * x0 + np.sqrt(1-alpha) * noise

    Args:
        df (pd.DataFrame): DataFrame with real data
        loc (float): Mean for white noise
        scale (float): Standard deviation for white noise
        t (np.array): Matrix with timesteps for each sample
        alpha_cumprod (np.cumprod): Compounded diffusion rates

    Returns (pd.DataFrame): Samples with noise, examples of column names [0, 1, Noise 0, Noise 1] where [0, 1] are 'df' after noise has been added and [Noise 0, Noise 1] are the noise added. 
    """

    df_noise, noise = add_noise(df, loc, scale, t, alpha_cumprod)

    # Add noise columns
    for col in noise.columns:
        noise_col_name = f'Noise {col}'
        df_noise[noise_col_name] = noise[col]

    return df_noise

if __name__=='__main__':
    infile = 'real_samples/gaussian_test.csv'
    outfile = 'gaussian_noise_test.csv'
    scale = 1   # standard deviation for white noise

    df = pd.read_csv(infile)
    
    df_noise = samples_with_noise(df, scale)

    df_noise.to_csv(f'noise_samples/{outfile}', index=False)