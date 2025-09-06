# generate_samples

Experimenting with generating samples. Is the distriution preserved? Are dependencies preserved?

## Procedure

1. Create and activate virtual environment. 
2. Install packages: pip install pandas matplotlib scipy fitter tensorflow scikit-learn

3. Go to generate/create_real_samples.py to modify distribution and dependencies within samples. Choose between uniform or normal distribution. 
4. Run python -m generate.create_real_samples to get the samples in a csv file. They are stored in /real_samples. These will act like real data. 

5. Go to generate/create_noise to modify what noise is being added. 
6. Run python -m generate.create_noise to get the transformed samples and the added noise in a csv file. They are stored in /noise_samples. These will be used in training the neural network.