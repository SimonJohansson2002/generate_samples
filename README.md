# generate_samples

Experimenting with generating samples. Is the distriution preserved? Are dependencies preserved?

## Procedure

1. Create and activate virtual environment. 
2. Install packages: pip install pandas matplotlib scipy fitter
3. Go to generate/create_real_samples.py to modify distribution and dependencies within samples. Choose between uniform or normal distribution. 
4. Run python -m generate.create_real_samples to get the samples in a csv file. They are stored in /real_samples. These will act like real data. 