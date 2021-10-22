import clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from scipy.stats import gamma

pd.set_option('display.max_rows', 100)

combined_deaths = 'combined_deaths'
date_col = 'Date'

def plot_posterior(a, b, week):
    # Take arbitrary X-axis range of (5, 20)
    x_points = np.linspace(5, 20, 500)
    # Generate PDF for gamma(a,b) for each x
    y_pdf = gamma.pdf(x_points, a=a, scale=b)

    map_val = round(x_points[y_pdf.argmax()], 3)
    print("MAP for week {0} Posterior Distribution of λ = {1}".format(week, map_val))
    plt.plot(x_points, y_pdf, label = 'Posterior: Week {0}, MAP = {1}'.format(week, map_val))

def calculate_posterior(daily_data):
    # Take a subset of only first 8 weeks of June-July 2020 data
    start_date, date_format = '2020-06-01', '%Y-%m-%d'
    end_date = (datetime.strptime(start_date, date_format) + timedelta(weeks=8)).strftime(date_format)
    bi_data = daily_data[(daily_data[date_col] >= start_date) & (daily_data[date_col] < end_date)].copy()
    # Compute combined deaths for the months of June-July
    bi_data[combined_deaths] = bi_data.loc[:, ['CT deaths', 'DC deaths']].sum(axis=1)

    # Given, combined deaths are assumed to be Poisson(lambda) distributed. MME for the Poisson parameter, lambda_mme, is the same as 
    # the sample mean of the first 4 weeks' data
    training_days = 4 * 7
    training_end_date = (datetime.strptime(start_date, date_format) + timedelta(days=training_days)).strftime(date_format)
    lambda_mme = bi_data[bi_data[date_col] < training_end_date][combined_deaths].mean()
    # Given, mean of prior exponential = beta = lambda_mme
    beta = lambda_mme

    plt.figure('Posterior Distribution', figsize=(15,8))
    print("{0} 2d) Posterior Distributions of λ {0}".format(20*"-"))
    # Find posterior from weeks 5-8
    for week in range(5, 9):
        week_end_date = (datetime.strptime(start_date, date_format) + timedelta(weeks=week)).strftime(date_format) 
        week_i_deaths = bi_data[bi_data[date_col] < week_end_date][combined_deaths]
        
        # As derived in the report, since both Poisson and Exponential distributions have Gamma distribution as their conjugate priors,
        # the posteriors too will be gamma distributed in the form of Gamma(a, b), where:
        # shape parameter a = sample sum sigma_x_i + 1, scale parameter b = 1 / (sample data size + (1/beta))
        sample_size = week_i_deaths.size
        a = week_i_deaths.sum() + 1
        b = 1 / (sample_size + (1/beta))
        plot_posterior(a, b, week)
    
    # Plot posterior distribution graph for weeks 5-8
    plt.title('Posterior Distribution of λ')
    plt.xlabel('X')
    plt.ylabel('λ ~ Gamma(a, b) PDF')
    plt.legend(loc='best')
    plt.show()
