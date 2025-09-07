from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras import layers, models
import tensorflow as tf
import pandas as pd
import numpy as np


def sigmoid(x: np.array) -> np.array:
    """
    Transforms values between (-inf, inf) into values between (0, 1) using sigmoid function.

    Args:
        x (np.array): Original values between (-inf, inf)

    Returns:
        np.array: Transformed values between (0, 1)
    """
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x: np.array) -> np.array:
    """
    Transforms values between (0, 1) to (-inf, inf) using inverse sigmoid function. 

    Args:
        x (np.array): Values between (0, 1).

    Returns:
        np.array: Transformed values between (-inf, inf).
    """
    return -np.log(1/x - 1)


def get_xy(df: pd.DataFrame) -> tuple[np.array]:
    """
    Extracts the input data and labels. 

    Args:
        df (pd.DataFrame): Dataframe where column with labels have 'Noise' in the name. 

    Returns:
        tuple[np.array]: X and y
    """
    columns = df.columns

    inputs = []
    labels = []

    for col in columns:
        if type(col) == str and 'Noise' in col:
            labels.append(col)
        else:
            inputs.append(col)
    
    X = df[inputs]
    y = df[labels]

    return np.array(X), np.array(y)


def train_nn(X: np.array, 
             y: np.array, 
             t: np.array, 
             T: int, 
             scaler: StandardScaler, 
             batch_size: int,
             epochs: int,
             model: models.Model = None,
             X_units: int = 16,
             t_units: int = 16,
             concatenate_units: list[int] = [16]) -> tuple[models.Model, float]:
    """
    Trains a timestep-conditioned neural network. Data does not need to be scaled.

    Args:
        X (np.array): Input data, shape (n_samples, n_features).
        y (np.array): Target noise, shape (n_samples, n_features).
        t (np.array): Timesteps for each sample, shape (n_samples,).
        T (int): Maximum number of timesteps.
        scaler (StandardScaler): Scaler for X.
        batch_size (int): Batch size when fitting data. 
        epochs (int): Number of epochs used in training. 
        model (models.Model, optional): Existing model to continue training. Defaults to None.
        X_units (int, optional): Number of neurons for X-values. 
        t_units (int, optional): Number of neurons for t-values. 
        concatenate_units (list[int], optional): Number of neurons for concatenated X- and t-values. If several values in the list, then layers are added. 

    Returns:
        tuple[models.Model, float]: Trained model and loss for first epoch.
    """
    X = X.astype('float32')
    y = y.astype('float32')
    t = t.astype('int32')

    if model is None:
        n_features = X.shape[1]

        # Define model using Functional API
        x_input = layers.Input(shape=(n_features,))
        t_input = layers.Input(shape=(), dtype=tf.int32)

        # Time embedding
        t_emb = layers.Embedding(input_dim=T+1, output_dim=32)(t_input)
        t_emb = layers.Dense(t_units, activation="relu")(t_emb)

        # Feature projection
        x_proj = layers.Dense(X_units, activation="relu")(x_input)

        # Combine
        h = layers.Concatenate()([x_proj, t_emb])
        for c in concatenate_units:
            h = layers.Dense(c, activation="relu")(h)

        output = layers.Dense(n_features)(h)

        model = models.Model(inputs=[x_input, t_input], outputs=output)
        model.compile(optimizer="adam", loss="mse")

        # Scale data
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Train the model
    history = model.fit([X_scaled, t], y, epochs=epochs, batch_size=batch_size, verbose=1)
    first_loss = history.history['loss'][0]

    return model, first_loss


def get_predicted_noise(X: np.array, t: np.array, model: models.Sequential, scaler: StandardScaler) -> pd.DataFrame:
    """
    Takes noisy data and predicts the noise.

    Args:
        X (np.array): Input data, shape (n_samples, n_features). Not scaled data.
        t (np.array): Matrix with timesteps for each sample.
        model (models.Sequential): Neural network model.
        scaler (StandardScaler): Scaler for X.

    Returns:
        pd.DataFrame: Predicted noise
    """
    # scale the samples using the same scaler as everywhere else
    X_scaled = scaler.transform(X)

    # predict the noise
    y_predicted = model.predict([X_scaled, t])

    return pd.DataFrame(y_predicted)