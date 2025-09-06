import matplotlib.pyplot as plt
from fitter import Fitter
from scipy import stats
import pandas as pd
import numpy as np

def plot_dist(df: pd.DataFrame, filename: str):
    """
    Sublot with one element per graph, e.g. first element in all samples in the first graph. Histograms, fitted distribution with parameters and Q-Q plots. Saves figure as svg.

    Args:
        df (pd.DataFrame): samples
        filename (str): name for image, e.g. 'imagename.svg'
    """

    n_cols = len(df.columns)

    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 4 * n_cols))

    if n_cols == 1:
        axes = np.array([axes])

    for i, col in enumerate(df.columns):
        data = df[col].dropna().values

        # ---- Histogram + best-fit distribution ----
        ax_hist = axes[i, 0]
        ax_hist.hist(data, bins=30, density=True, alpha=0.6, color="skyblue", edgecolor="black")

        # Fit distribution using fitter
        f = Fitter(data)
        f.fit()
        best_dist = list(f.get_best().keys())[0]
        params = f.fitted_param[best_dist]

        # Get scipy distribution object
        dist = getattr(stats, best_dist)

        # Overlay fitted pdf
        x = np.linspace(min(data), max(data), 200)
        pdf = dist.pdf(x, *params)
        ax_hist.plot(x, pdf, "r-", lw=2, label=f"{best_dist}\n{params}")
        ax_hist.set_title(f"{col}: Histogram + Fit")
        ax_hist.legend()

        # ---- Q-Q plot ----
        ax_qq = axes[i, 1]
        stats.probplot(data, dist=dist, sparams=params, plot=ax_qq)
        ax_qq.set_title(f"{col}: Q-Q Plot")

    plt.tight_layout()
    fig.savefig(f'images/{filename}', format="svg")
    plt.close(fig)

if __name__=='__main__':
    infile = 'real_samples/gaussian.csv'
    outfile = 'gaussian.svg'

    df = pd.read_csv(infile, index_col=0)  # don’t treat index as a column

    plot_dist(df, outfile)
