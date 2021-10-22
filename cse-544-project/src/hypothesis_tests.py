import math
import pandas as pd
from scipy.stats import norm, t
# pd.set_option('display.max_rows', 100)

date_col = 'Date'
CT_confirmed = 'CT confirmed'
DC_confirmed = 'DC confirmed'
CT_deaths = 'CT deaths'
DC_deaths = 'DC deaths'

def walds_test_one_sample(feb_state_data, march_state_data, hypothesis_type, state):
    # theta_o, assumed true mean calculated over the Feb'21 data
    theta_o = feb_state_data.sum() / feb_state_data.size
    # theta_cap, sample mean calculated over the March'21 data = sigma_X_i / sample_size
    theta_cap = march_state_data.sum() / march_state_data.size
    
    # Since we are given that the daily data is Poisson distributed, from A4Q3, we know that lambda_mle is same as the sample mean (theta_cap here)
    lambda_mle = theta_cap
    # For Poisson distribution, variance = lambda. Hence, we can plugin the estimate lambda_mle to be the sample variance for the March data.
    sample_variance = lambda_mle
    # the std_error estimate of lambda_mle for March values will then be same as sqrt(sample_variance / sample_size_march)
    se_cap_lambda_cap = math.sqrt(sample_variance / march_state_data.size)

    # Calculate W statistic = (theta_cap - theta_o) / std_error_cap
    W = round((theta_cap - theta_o) / se_cap_lambda_cap, 3)

    # Accept/Reject the null hypothesis based on W and the critical value
    alpha = 0.05
    # z_alpha_by_2 = 1.96 for alpha = 0.05
    critical_value = round(norm.ppf(1 - (alpha / 2)), 2)

    print("Wald's 1-Sample Test Result for comparing means of daily {0} in {1} across Feb'21 and March'21".format(hypothesis_type, state))
    print("Sample mean = {0:.3f}, True mean = {1:.3f}, std error estimate = {2:.3f}".format(theta_cap, theta_o, se_cap_lambda_cap))
    if abs(W) > critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as W-statistic = {2} exceeds threshold {3}\n".format(hypothesis_type, state, abs(W), critical_value))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as W-statistic = {2} does not exceed threshold {3}\n".format(hypothesis_type, state, abs(W), critical_value))
    
def walds_test_two_sample(feb_state_data, march_state_data, hypothesis_type, state):
    n, m = feb_state_data.size, march_state_data.size
    X_mean = feb_state_data.sum() / n
    Y_mean = march_state_data.sum() / m

    # Similar to the 1-Sample Test, we calculate the std error estimate as sqrt(var(X)/n + Var(Y)/m). Since the data is given to
    # be Poisson distributed, Var(X) = lambda_mle = sample mean
    se_cap = math.sqrt((X_mean / n) + (Y_mean / m))

    # Calculate W statistic = (X_bar - Y_bar) / std_error_cap
    W = round((X_mean - Y_mean) / se_cap, 3)

    # Accept/Reject the null hypothesis based on W and the critical value
    alpha = 0.05
    # z_alpha_by_2 = 1.96 for alpha = 0.05
    critical_value = round(norm.ppf(1 - (alpha / 2)), 2)

    print("Wald's 2-Sample Test Result for comparing means of daily {0} in {1} across Feb'21 and March'21".format(hypothesis_type, state))
    print("Feb Mean = {0:.3f}, March Mean = {1:.3f}, std error estimate = {2:.3f}".format(X_mean, Y_mean, se_cap))
    if abs(W) > critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as W-statistic = {2} exceeds threshold {3}\n".format(hypothesis_type, state, abs(W), critical_value))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as W-statistic = {2} does not exceed threshold {3}\n".format(hypothesis_type, state, abs(W), critical_value))

def z_test_one_sample(feb_state_data, march_state_data, state_data_all, hypothesis_type, state):
    # mu_o, assumed true mean calculated over the Feb'21 data
    mu_o = feb_state_data.sum() / feb_state_data.size
    # mu_cap, sample mean calculated over the March'21 data = sigma_X_i / sample_size
    sample_size = march_state_data.size
    sample_mean = march_state_data.sum() / sample_size

    # Calculate true std dev/err over the entire data per #cases/#deaths columns for each state. This will be same as the corrected sample std error
    # over the entire covid dataset given to us, i.e, sigma = sqrt(sum((X_i - X_bar)^2) / (n-1))
    sigma_true = math.sqrt(((state_data_all - state_data_all.mean()) ** 2).sum() / (state_data_all.size - 1))

    # Calculate Z statistic = (sample_mean - mu_o) / (sigma / sqrt(sample_size)))
    Z = round((sample_mean - mu_o) / (sigma_true / math.sqrt(sample_size)), 3)

    # Accept/Reject the null hypothesis based on Z and the critical value
    alpha = 0.05
    # z_alpha_by_2 = 1.96 for alpha = 0.05
    critical_value = round(norm.ppf(1 - (alpha / 2)), 2)

    print("1-Sample Z Test Result for comparing means of daily {0} in {1} across Feb'21 and March'21".format(hypothesis_type, state))
    print("Sample mean = {0:.3f}, True mean = {1:.3f}, True std deviation = {2:.3f}".format(sample_mean, mu_o, sigma_true))
    if abs(Z) > critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as Z-statistic = {2} exceeds threshold {3}\n".format(hypothesis_type, state, abs(Z), critical_value))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as Z-statistic = {2} does not exceed threshold {3}\n".format(hypothesis_type, state, abs(Z), critical_value))

def t_test_one_sample(feb_state_data, march_state_data, hypothesis_type, state):
    # mu_o, assumed true mean calculated over the Feb'21 data
    mu_o = feb_state_data.sum() / feb_state_data.size
    # mu_cap, sample mean calculated over the March'21 data = sigma_X_i / sample_size
    sample_size = march_state_data.size
    sample_mean = march_state_data.sum() / sample_size

    # Calculate corrected sample std dev/err = sqrt(sum((sample_i - sample_mean)^2) / (sample_size-1))
    sample_std_dev = math.sqrt(((march_state_data - sample_mean) ** 2).sum() / (sample_size - 1))

    # Calculate Z statistic = (sample_mean - mu_o) / (sample_std_dev / sqrt(sample_size)))
    T = round((sample_mean - mu_o) / (sample_std_dev / math.sqrt(sample_size)), 3)
    
    # Accept/Reject the null hypothesis based on Z and the critical value
    # Degrees of freedom for 1-Sample test = sample_size - 1
    df_t = sample_size - 1
    alpha = 0.05
    # t_n_minus_1_alpha_by_2 = 2.042 for n = 31, alpha = 0.05
    critical_value = round(t.ppf(1 - (alpha / 2), df=df_t), 3)

    print("1-Sample T Test Result for comparing means of daily {0} in {1} across Feb'21 and March'21".format(hypothesis_type, state))
    print("Sample mean = {0:.3f}, True mean = {1:.3f}, Sample std deviation = {2:.3f}".format(sample_mean, mu_o, sample_std_dev))
    if abs(T) > critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as T-statistic = {2} exceeds threshold {3}\n".format(hypothesis_type, state, abs(T), critical_value))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as T-statistic = {2} does not exceed threshold {3}\n".format(hypothesis_type, state, abs(T), critical_value))

def t_test_two_sample_unpaired(feb_state_data, march_state_data, hypothesis_type, state):
    n, m = feb_state_data.size, march_state_data.size
    X_mean = feb_state_data.sum() / n
    Y_mean = march_state_data.sum() / m

    # Calculate corrected sample variances of both Feb(X) and March(Y) samples
    X_var = math.sqrt(((feb_state_data - X_mean) ** 2).sum() / (n - 1))
    Y_var = math.sqrt(((march_state_data - Y_mean) ** 2).sum() / (m - 1))
    # sample pool standard deviation = sqrt((sample_var_X/n) + (sample_var_Y/m))
    sample_pool_std_dev = math.sqrt((X_var / n) + (Y_var / m))

    # T-Statistic for 2 sample test is given by T = diff in sample mean / sample pool standard deviation
    T = round((X_mean - Y_mean) / sample_pool_std_dev, 3)

    # Accept/Reject the null hypothesis based on Z and the critical value
    # Degrees of freedom for unpaired test = (n-1) + (m-1)
    df = n + m - 2
    alpha = 0.05
    # t_df_alpha_by_2 = 2.002 for n = 28, m = 31, alpha = 0.05
    critical_value = round(t.ppf(1 - (alpha / 2), df=df), 3)

    print("Unpaired 2-Sample T Test Result for comparing means of daily {0} in {1} across Feb'21 and March'21".format(hypothesis_type, state))
    print("Feb Mean = {0:.3f} with sample size {1}, March Mean = {2:.3f} with sample size {3}, Sample pool std deviation = {4:.3f}".format(X_mean, n, Y_mean, m, sample_pool_std_dev))
    if abs(T) > critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as T-statistic = {2} exceeds threshold {3}\n".format(hypothesis_type, state, abs(T), critical_value))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the mean of daily {0} is same for Feb’21 and March’21 in {1}, "
        "as T-statistic = {2} does not exceed threshold {3}\n".format(hypothesis_type, state, abs(T), critical_value))

def run_hypothesis_tests(daily_data):
    feb_21_range = (daily_data[date_col] >= '2021-02-01') & (daily_data[date_col] <= '2021-02-28')
    feb_21_cases_CT = daily_data[feb_21_range][CT_confirmed].to_numpy()
    feb_21_cases_DC = daily_data[feb_21_range][DC_confirmed].to_numpy()
    feb_21_deaths_CT = daily_data[feb_21_range][CT_deaths].to_numpy()
    feb_21_deaths_DC = daily_data[feb_21_range][DC_deaths].to_numpy()
    
    mar_21_range = (daily_data[date_col] >= '2021-03-01') & (daily_data[date_col] <= '2021-03-31')
    mar_21_cases_CT = daily_data[mar_21_range][CT_confirmed].to_numpy()
    mar_21_cases_DC = daily_data[mar_21_range][DC_confirmed].to_numpy()
    mar_21_deaths_CT = daily_data[mar_21_range][CT_deaths].to_numpy()
    mar_21_deaths_DC = daily_data[mar_21_range][DC_deaths].to_numpy()
    
    print("\n{0} 2b) Wald's, Z and T Hypotheses Tests {0}".format(20*"-"))
    # In all of the 1-Sample tests below, we are considering March data for running the tests, and Feb data for calculating the assumed true mean
    print("{0} Wald's 1-Sample Test {0}".format(20*"-"))
    # 1-Sample Wald's Test for confirmed cases in CT and DC
    walds_test_one_sample(feb_21_cases_CT, mar_21_cases_CT, 'confirmed positive cases', 'CT')
    walds_test_one_sample(feb_21_cases_DC, mar_21_cases_DC, 'confirmed positive cases', 'DC')
    # 1-Sample Wald's Test for deaths in CT and DC
    walds_test_one_sample(feb_21_deaths_CT, mar_21_deaths_CT, 'deaths', 'CT')
    walds_test_one_sample(feb_21_deaths_DC, mar_21_deaths_DC, 'deaths', 'DC')

    print("{0} 1-Sample Z Test {0}".format(20*"-"))
    # 1-Sample Z Test for confirmed cases in CT and DC
    z_test_one_sample(feb_21_cases_CT, mar_21_cases_CT, daily_data[CT_confirmed].to_numpy(), 'confirmed positive cases', 'CT')
    z_test_one_sample(feb_21_cases_DC, mar_21_cases_DC, daily_data[DC_confirmed].to_numpy(), 'confirmed positive cases', 'DC')
    # 1-Sample Z Test for deaths in CT and DC
    z_test_one_sample(feb_21_deaths_CT, mar_21_deaths_CT, daily_data[CT_deaths].to_numpy(), 'deaths', 'CT')
    z_test_one_sample(feb_21_deaths_DC, mar_21_deaths_DC, daily_data[DC_deaths].to_numpy(), 'deaths', 'DC')

    print("{0} 1-Sample T Test {0}".format(20*"-"))
    # 1-Sample T Test for confirmed cases in CT and DC
    t_test_one_sample(feb_21_cases_CT, mar_21_cases_CT, 'confirmed positive cases', 'CT')
    t_test_one_sample(feb_21_cases_DC, mar_21_cases_DC, 'confirmed positive cases', 'DC')
    # 1-Sample T Test for deaths in CT and DC
    t_test_one_sample(feb_21_deaths_CT, mar_21_deaths_CT, 'deaths', 'CT')
    t_test_one_sample(feb_21_deaths_DC, mar_21_deaths_DC, 'deaths', 'DC')

    print("{0} Wald's 2-Sample Test {0}".format(20*"-"))
    # 2-Sample Wald's Test for confirmed cases in CT and DC
    walds_test_two_sample(feb_21_cases_CT, mar_21_cases_CT, 'confirmed positive cases', 'CT')
    walds_test_two_sample(feb_21_cases_DC, mar_21_cases_DC, 'confirmed positive cases', 'DC')
    # 2-Sample Wald's Test for deaths in CT and DC
    walds_test_two_sample(feb_21_deaths_CT, mar_21_deaths_CT, 'deaths', 'CT')
    walds_test_two_sample(feb_21_deaths_DC, mar_21_deaths_DC, 'deaths', 'DC')

    print("{0} Unpaired 2-Sample T Test {0}".format(20*"-"))
    # 1-Sample T Test for confirmed cases in CT and DC
    t_test_two_sample_unpaired(feb_21_cases_CT, mar_21_cases_CT, 'confirmed positive cases', 'CT')
    t_test_two_sample_unpaired(feb_21_cases_DC, mar_21_cases_DC, 'confirmed positive cases', 'DC')
    # 1-Sample T Test for deaths in CT and DC
    t_test_two_sample_unpaired(feb_21_deaths_CT, mar_21_deaths_CT, 'deaths', 'CT')
    t_test_two_sample_unpaired(feb_21_deaths_DC, mar_21_deaths_DC, 'deaths', 'DC')
