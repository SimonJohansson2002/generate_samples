from sklearn.metrics import mean_squared_error
from keras import layers, models
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


def train_nn(X: np.array, y: np.array, model: models.Sequential = None) -> models.Sequential:
    """
    Trains a neural network based on X and y. Option to train already existing neural network. 

    Args:
        X (np.array): Input data. 
        y (np.array): Output/labels.
        model (models.Sequential, optional): Existing neural network. Defaults to None. 

    Returns:
        models.Sequential: neural network model
    """

    X = X.astype('float32')
    y = y.astype('float32')

    if not model:
        n_features = X.shape[1]

        # Build a simple fully-connected neural network
        model = models.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(n_features)  # output dimension same as input/noise
        ])

        model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    return model


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

    df = pd.read_csv(infile)
    X, y = get_xy(df)
    model = train_nn(X, y)

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