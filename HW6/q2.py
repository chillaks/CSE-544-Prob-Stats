import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_posteriors(x, y_square, sigma, i):
    y = math.sqrt(y_square)
    # Range of points on the x-axis from μ - 4σ to μ + 4σ
    x_points = np.arange(x - 4 * y, x + 4 * y, 0.01)
    # theta ~ Normal(x, y^2)
    theta_pdf = norm.pdf(x_points, loc = x, scale = y)
    plt.plot(x_points, theta_pdf, label = 'Posterior: Iteration ' + str(i))

def calculate_posterior(row_data, sigma):
    # Starting with Standard Normal as prior distribution. So mu=a=0, variance=b^2=1
    a, b_square = 0, 1

    plt.figure('Posterior Distributions', figsize=(20,8))
    result_df = pd.DataFrame(columns=['Mean_hat', 'Variance_hat'])
    for i in range(row_data.shape[0]):
        row = row_data.iloc[i].values
        # number of data points per row
        n = row.shape[0]
        # given, std_dev = sigma^2 / n
        se_square = (sigma ** 2) / n
        # theta_mean x = (b^2 * X_bar + se^2 * a) / (b^2 + se^2)
        x = (b_square * row.mean() + se_square * a) / (b_square + se_square)
        # theta_var y^2 = (b^2 * se^2) / (b^2 + se^2)
        y_square = (b_square * se_square) / (b_square + se_square)
        result_df.loc[i] = [x, y_square]

        # Plot posterior curve for current iteration
        plot_posteriors(x, y_square, sigma, i+1)
        # The posterior of current(ith) iteration will be used as the prior to calculate posterior of next(i+1th) iteration
        a, b_square = x, y_square

    result_df.index += 1
    print(result_df)
    # Plot posterior distribution graph for all 5 iterations
    plt.title('Posterior Distributions for sigma = ' + str(sigma))
    plt.xlabel('X')
    plt.ylabel('θ ~ Normal(x, y^2) PDF')
    plt.legend(loc='upper left')
    plt.show()

# calculate posterior over 5 iterations for sigma=3
print('----------------------------------------- PART a) --------------------------------------')
dataset_sigma3 = pd.read_csv('./datasets/q2_sigma3.dat', header=None)
sigma = 3
calculate_posterior(dataset_sigma3, sigma)

# calculate posterior over 5 iterations for sigma=100
print('\n----------------------------------------- PART b) --------------------------------------')
dataset_sigma3 = pd.read_csv('./datasets/q2_sigma100.dat', header=None)
sigma = 100
calculate_posterior(dataset_sigma3, sigma)