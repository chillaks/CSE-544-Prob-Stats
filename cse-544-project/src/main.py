from datetime import date
import pandas as pd

import clean
import auto_regression
import ewma
import one_sample_ks_perm
from ks_test import KS_2_Sample_Test
from hypothesis_tests import run_hypothesis_tests
from posterior import calculate_posterior

from pearson_correlation_test import perform_pearsons_correlation_test
import chi_square
import exploratory

# A) Mandatory tasks to be performed on assigned COVID-19 dataset (4.csv)
print("{0} A) Mandatory Tasks on assigned COVID-19 dataset (4.csv) {0}".format(30*"-"))
# 1) Clean dataset and detect outliers using Tukey’s rule. Also split given cumulative data into daily #cases/#deaths
data, daily_data = clean.get_cleaned_data("../data/States Data/4.csv", drop_outliers=False)
data['Date'] = pd.to_datetime(data['Date'])
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
#print(data)

# 2a) Time Series analysis
print("\n{0} 2a) Time Series Analysis {0}".format(20*"-"))
auto_regression.perform_auto_regression(daily_data, 3)
auto_regression.perform_auto_regression(daily_data, 5)
ewma.run_ewma_analysis(daily_data)

# 2b) Wald's, Z and T Tests on the #cases/#deaths data of the 2 states in the given time range
run_hypothesis_tests(daily_data)

# 2c) Perform 1/2-Sample KS and Permutations tests on the #cases/#deaths data of the 2 states
one_sample_ks_perm.KS_1_sample_main(daily_data)
one_sample_ks_perm.Permutation_main(daily_data)
KS_2_Sample_Test(daily_data, 'confirmed')
KS_2_Sample_Test(daily_data, 'deaths')

# 2d) Apply Bayesian Inference to calculate the posterior for combined deaths data
calculate_posterior(daily_data)

# B) Exploratory tasks to be performed using US-all and X datasets. We have chosen our X dataset to be US domestic Flights cancellation data from Jan-Jun 2020.
# Full dataset can be found at https://www.kaggle.com/akulbahl/covid19-airline-flight-delays-and-cancellations?select=jantojun2020.csv
print("\n{0} B) Exploratory Tasks using US-all Covid and X (US Domestic Flight Cancellations) datasets {0}".format(30*"-"))
_, us_all_daily_data = clean.get_cleaned_data("../data/US-all/US_confirmed.csv", us_all=True, drop_outliers=False)

# Since our X dataset has flight data only from Jan-Jun 2020, and the US-all dataset has sparse data for Jan 2020 (just 9 entries),
# we will restrict our tests to only use Feb-June subset of US-all covid data.
monthly_cases_mean = exploratory.monthly_mean_daily_cases(us_all_daily_data, '2020-02-01', '2020-06-30')

# Find months with min and max monthly cases in New York, which we will use later in the hypostheses for our inferences.
# Both will be of the form '<month_number> <year>'. Eg: min='3 2020' denotes the month with least average cases is March 2020.
min_month_NY, max_month_NY = monthly_cases_mean['NY'].idxmin(), monthly_cases_mean['NY'].idxmax()

# Read the X dataset, which will be the US flights data for all states from Jan-Jun 2020
flights_data = pd.read_csv("../data/X_flights_cancellation/jantojun2020.csv")
# Filter only the flights departing from / arriving to New York
flights_data_NY = flights_data[(flights_data['ORIGIN_STATE_ABR'] == 'NY') | (flights_data['DEST_STATE_ABR'] == 'NY')]

# Inference 1: We perform pearson's correlation test to measure the correlation between number of cases and number of flight cancellations for the
# state of New York in the range of Jan to Jun 2020
perform_pearsons_correlation_test('NY', date(day=22, month=1, year=2020), date(day=30, month=6, year=2020), flights_data_NY, us_all_daily_data[['Date','NY']])

# Inference 2: We perform a chi-square test to check whether the presence of covid cases affected the number of flight cancellations.
# We take the count of flights cancelled in the months with lowest, and highest covid cases, and use them to test our hypothesis.
chi_square.perform_chi_square_test(min_month_NY, max_month_NY, flights_data)

# Inference 3: Using T-Test to determine if covid-19 may have had an impact on daily domestic flight cancellation by comparing the means of
# cancellations in a month where there is minimum/no covid cases with a month having the highest average daily cases
exploratory.one_tailed_unpaired_t_test(flights_data_NY, min_month_NY, max_month_NY)
