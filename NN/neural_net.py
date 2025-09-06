from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras import layers, models
import tensorflow as tf
import pandas as pd
import numpy as np

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
        if 'Noise' in col:
            labels.append(col)
        else:
            inputs.append(col)
    
    X = df[inputs]
    y = df[labels]

    return np.array(X), np.array(y)


def train_nn(X: np.array, y: np.array, t: np.array, T: int, scaler: StandardScaler, model: models.Model = None) -> models.Model:
    """
    Trains a timestep-conditioned neural network.

    Args:
        X (np.array): Input data, shape (n_samples, n_features).
        y (np.array): Target noise, shape (n_samples, n_features).
        t (np.array): Timesteps for each sample, shape (n_samples,).
        T (int): Maximum number of timesteps.
        model (models.Model, optional): Existing model to continue training. Defaults to None.

    Returns:
        models.Model: Trained model.
    """
    X = X.astype('float32')
    y = y.astype('float32')
    t = t.astype('int32')

    X_scaled = scaler.fit_transform(X)

    if model is None:
        n_features = X.shape[1]

        # Define model using Functional API
        x_input = layers.Input(shape=(n_features,))
        t_input = layers.Input(shape=(), dtype=tf.int32)

        # Time embedding
        t_emb = layers.Embedding(input_dim=T+1, output_dim=32)(t_input)
        t_emb = layers.Dense(128, activation="relu")(t_emb)

        # Feature projection
        x_proj = layers.Dense(128, activation="relu")(x_input)

        # Combine
        h = layers.Concatenate()([x_proj, t_emb])
        h = layers.Dense(128, activation="relu")(h)
        output = layers.Dense(n_features)(h)

        model = models.Model(inputs=[x_input, t_input], outputs=output)
        model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit([X_scaled, t], y, epochs=50, batch_size=32, verbose=1)

    return model, scaler


def get_predicted_error(df: pd.DataFrame, model: models.Sequential) -> pd.DataFrame:
    """
    Takes noisy data and predicts the noise.

    Args:
        df (pd.DataFrame): Noisy data, must have the same shape as the model input data. The columns with labels have 'Noise' in the name.
        model (models.Sequential): Neural network model

    Returns:
        pd.DataFrame: predicted values
    """

    columns = df.columns

    inputs = []

    for col in columns:
        if 'Noise' in col:
            continue
        else:
            inputs.append(col)
    
    X = np.array(df[inputs])

    y_predicted = model.predict(X)

    pred_columns = [f"Pred Noise {i}" for i in inputs]

    return pd.DataFrame(y_predicted, columns=pred_columns)

if __name__=='__main__':
    infile = 'noise_samples/gaussian_noise.csv'
    testfile = 'noise_samples/gaussian_noise_test.csv'
    scale = 1
    size = 10
    t = 1000
    T = 1000

    scaler = StandardScaler()
    df = pd.read_csv(infile)
    X, y = get_xy(df)
    model = train_nn(X, y, t, T, scaler)

    df_test = pd.read_csv(testfile)
    y_pred = get_predicted_error(df_test, model)

    columns = df_test.columns
    labels = []

    for col in columns:
        if 'Noise' in col:
            labels.append(col)
        else:
            continue

    mse = mean_squared_error(df_test[labels].values.flatten(), y_pred.values.flatten())

    print(mse)