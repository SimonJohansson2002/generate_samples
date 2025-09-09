from sklearn.preprocessing import StandardScaler
from .create_noise import get_noise
from keras import models
import pandas as pd
import numpy as np


def ddpm(X: pd.DataFrame, 
         t: int, 
         noise_pred: pd.DataFrame, 
         alphas: np.ndarray, 
         alpha_cumprod: np.ndarray, 
         betas: np.ndarray, 
         noise_loc: float = 0, 
         noise_scale: float = 1) -> pd.DataFrame:
    
    alpha = alphas[t]
    alpha_compound = alpha_cumprod[t]

    if t == 0:
        X_prev = 1 / np.sqrt(alpha) * (X - (1-alpha) / np.sqrt(1-alpha_compound) * noise_pred)
    else:
        s = np.sqrt(betas[t])
        noise = get_noise(X, loc=noise_loc, scale=noise_scale)
        X_prev = 1 / np.sqrt(alpha) * (X - (1-alpha) / np.sqrt(1-alpha_compound) * noise_pred) + s * noise

    return X_prev

def diffusion(X: pd.DataFrame, 
              T: int, 
              model: models.Model, 
              scaler: StandardScaler, 
              alphas: np.ndarray, 
              alpha_cumprod: np.ndarray, 
              betas: np.ndarray, 
              noise_loc: float = 0, 
              noise_scale: float = 1) -> pd.DataFrame:
    
    t = T-1
    n_rows, n_cols = X.shape

    while t >= 0:
        print(f'{T-t}/{T}')
        t_vec = np.full((n_rows,), t, dtype=np.int32)
        X_scaled = scaler.transform(np.array(X))
        noise_pred = model.predict([X_scaled, t_vec])

        X = ddpm(X=X,
                 t=t, 
                 noise_pred=noise_pred,
                 alphas=alphas,
                 alpha_cumprod=alpha_cumprod,
                 betas=betas,
                 noise_loc=noise_loc,
                 noise_scale=noise_scale)
        
        t -= 1
    
    return X