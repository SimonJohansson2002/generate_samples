from NN.neural_net import get_xy, train_nn, get_predicted_noise, sigmoid, inverse_sigmoid
from generate.create_real_samples import normal_dist, uni_dist
from generate.create_noise import samples_with_noise
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statistics.view import plot_dist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def func(xi: float) -> list:
    return [xi, xi*2]


def build_model(iterations: int = 1000,
                sample_size: int = 10000,
                loc_real: float = 0, 
                scale_real: float = 1,
                loc_noise: float = 0, 
                scale_noise: float = 1, 
                T: int = 999, 
                beta_start: float = 1e-4, 
                beta_end: float = 0.02, 
                epochs: int = 1,
                batch_size: int = 128,
                X_units: int = 32,
                t_units: int = 32,
                concatenate_units: int = 16):
    
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_cumprod = np.cumprod(alphas)

    model = None

    # plot for mse
    x_values = []
    y_test_loss = []
    y_train_loss = []

    scaler = StandardScaler()   # use same scaler throughout all training and testing to keep distribution

    for i in range(iterations):
        # create real samples
        real_samples = normal_dist(loc=loc_real, scale=scale_real, size=sample_size, function=func)

        # randomly pick a t from (0,T] for each sample
        rows, cols = real_samples.shape
        t = np.random.randint(0, T, size=(rows,))

        # create noise with t
        noise_samples = samples_with_noise(real_samples, loc=loc_noise, scale=scale_noise, t=t, alpha_cumprod=alpha_cumprod)
        
        # train the model
        X, y = get_xy(noise_samples)
        model, last_loss = train_nn(X=X, y=y, t=t, T=T, scaler=scaler, batch_size=batch_size, epochs=epochs, model=model, X_units=X_units, t_units=t_units, concatenate_units=concatenate_units)
        y_train_loss.append(last_loss)

        # create test
        t_test = np.random.randint(0, T, size=(rows,))
        test_noise_samples = samples_with_noise(real_samples, loc=loc_noise, scale=scale_noise, t=t_test, alpha_cumprod=alpha_cumprod)
        X_test, y_test = get_xy(test_noise_samples)
        predicted_noise = get_predicted_noise(X=X_test, t=t_test, model=model, scaler=scaler)
        mse = mean_squared_error(y_true=y_test, y_pred=predicted_noise)
        x_values.append(i)
        y_test_loss.append(mse)
    
    plt.plot(x_values, y_test_loss, label='Test loss')
    plt.plot(x_values, y_train_loss, label='Train loss')
    plt.legend()
    plt.title('MSE')
    plt.show()

    # create a random white noise sample
    # predict the noise
    # reverse the noise into samples, X_generated = scaler.inverse_transform(X_generated_scaled)
    # get statistics for the generated samples


if __name__=='__main__':
    build_model()