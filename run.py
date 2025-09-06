from NN.neural_net import get_xy, train_nn, get_predicted_error
from generate.create_real_samples import normal_dist, uni_dist
from generate.create_noise import samples_with_noise
from sklearn.metrics import mean_squared_error
from statistics.view import plot_dist
import pandas as pd
import numpy as np


def func():
    pass


def build_model(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_cumprod = np.cumprod(alphas)
    
    # start a while loop based on mse
    # create real samples
    # randomly pick a t from (0,T]
    t = np.random.randint(0, T)
    # create noise with t
    # train the model (with t as second input?)
    # get mse and continue looping until mse is below a threshold

    # create a random white noise sample
    # predict the noise
    # reverse the noise into samples, X_generated = scaler.inverse_transform(X_generated_scaled)
    # get statistics for the generated samples
    pass


if __name__=='__main__':
    build_model()