# generate_samples

Experimenting with generating samples through diffusion. Is the distriution preserved? Are dependencies preserved?

## 📂 Directory Overview

project-root/         | Description\
├── fake_samples/     | Stores generated synthetic samples\
├── generate/         | Scripts for generating real and synthetic samples\
├── images/           | Stores images and visual outputs\
├── NN/               | Neural network model implementation\
├── noise_samples/    | Stores real samples with added noise\
├── real_samples/     | Stores raw real samples\
├── statistics/       | Scripts for fitting models and plotting statistics\
├── README.md         | Project documentation and guides\
└── run.py            | Main script for training and testing the model

## Run.py

The idea is to generate "real" samples with known distribution and correlation within each sample and then to generate synthetic (or "fake") samples with similar properties. The structure of each sample is designed in the function 'func'. 

This procedure is done with diffusion. We add white noise to the "real" samples and train a neural network do predict the noise added. We create a stochastic process (forward diffusion process, DDPM):

$X_1 = \sqrt{\alpha_0} \cdot X_0 + \sqrt{1 - \alpha_0} \cdot \epsilon_0$

$X_2 = \sqrt{\alpha_1} \cdot X_1 + \sqrt{1 - \alpha_1} \cdot \epsilon_1$

...

$X_t = \sqrt{\alpha_{\text{t-1}}} \cdot X_{\text{t-1}} + \sqrt{1 - \alpha_{\text{t-1}}} \cdot \epsilon_{\text{t-1}}$

where

$X_0$ is a matrix of "real" samples,

$t \in {0, 1, 2,..., T}$, T is a fixed maximum timestep,

$\epsilon_t \sim \mathcal{N}(0, I)$,

$\alpha_t \in (0, 1)$ is a decaying coefficient.

Compounded this becomes

$X_t = \sqrt{\overline\alpha_{\text{t-1}}} \cdot X_0 + \sqrt{1 - \overline\alpha_{\text{t-1}}} \cdot \epsilon$

where

$\overline\alpha_t \in (0, 1)$ is the compounded value $\prod_{i=0}^{t-1} \alpha_i$.

This is used to generate the "real" samples with added noise. We feed $x_t$ together with $t$ to the neural network and predict the last added white noise $\epsilon$. Training is stopped after Mean Square Error (MSE) fall below a threshold or after a maximum number of iterations. Both are adjustable. 

After training, new white noise is generated as row vectors with the same dimension as the "real" samples. We set $t=T$ and feed into the neural network to predict the noise. It is then possible to reverse the calculations shown previously by iterating the DDPM uptade formula:

$X_{t-1} = \frac{1}{\sqrt{\alpha_t}} \Bigg( X_t - \frac{1 - \alpha_t}{\sqrt{1 - \overline\alpha_t}} \, \epsilon_{\text{pred}} \Bigg) + \sqrt{\beta_t} \, \epsilon$

where

$\epsilon_{pred}$ is the predicted noise and

$\beta_t = 1 - \alpha_t$ determines the variance of the stochastic sampling. This term is included when $t>1$ and zero when $t=1$.

When reaching $X_0$ we have the synthetic samples with a distribution close to that of the "real" samples. Some discrepancy may remain due to $\text{MSE} \neq 0$. Heuristically, larger MSE can sometimes lead to greater variance and deviations in the synthetic samples.