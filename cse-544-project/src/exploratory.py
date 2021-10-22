import math
import pandas as pd
from scipy.stats import t

# Columns present in dataset
date_col_us_all = 'Date'
date_col_flights = 'FL_DATE'
day_col_flights = 'DAY_OF_MONTH'
month_col_flights = 'MONTH'
year_col_flights = 'YEAR'
cancelled = 'CANCELLED'

# derived columns using existing columns present in dataset
num_flights = 'NUM_FLIGHTS'
num_cancelled = 'NUM_CANCELLED'
cancelled_ratio = 'CANCELLED_RATIO'

# Returns datadrame containing mean daily cases for every month, for each state
def monthly_mean_daily_cases(daily_data, l_date_range, u_date_range):
    state_data = daily_data[(daily_data[date_col_us_all] >= l_date_range) & (daily_data[date_col_us_all] <= u_date_range)].copy()
    
    # Use aggregation to find mean monthly cases for every state
    state_data[date_col_us_all] = pd.to_datetime(state_data[date_col_us_all])
    monthly_cases_mean = state_data.groupby(pd.Grouper(key=date_col_us_all, freq='1M')).mean()
    monthly_cases_mean.index = monthly_cases_mean.index.strftime('%m %Y')
    
    return monthly_cases_mean

# Perform One-Tailed Unpaired T-Test to accept or reject the null hypothesis H0, such that 
# H0: mean(daily cancellation ratio in month with no/least covid cases) >= mean(daily cancellation ratio in month with highest covid cases)
# H1: mean(daily cancellation ratio in month with no/least covid cases) < mean(daily cancellation ratio in month with highest covid cases)
def one_tailed_unpaired_t_test(flights_data_NY, min_month_NY, max_month_NY):
    # X - Data for month with least average daily covid cases
    min_month, min_year = int(min_month_NY.split(' ')[0]), int(min_month_NY.split(' ')[1])
    X_data = flights_data_NY[(flights_data_NY[month_col_flights] == min_month) & (flights_data_NY[year_col_flights] == min_year)].copy()
    X_data[date_col_flights] = pd.to_datetime(X_data[date_col_flights])
    # Compute ratio of daily cancellations, that is, num(cancelled flights) / num(total flights) per day for X data
    X_aggr = X_data.groupby([date_col_flights, day_col_flights, year_col_flights, month_col_flights]).size().reset_index(name=num_flights)
    X_aggr[num_cancelled] = X_data.groupby(date_col_flights)[cancelled].sum().reset_index(drop=True)
    X_aggr[cancelled_ratio] = X_aggr[num_cancelled] / X_aggr[num_flights]

    # Y - Data for month with highest average daily covid cases
    max_month, max_year = int(max_month_NY.split(' ')[0]), int(max_month_NY.split(' ')[1])
    Y_data = flights_data_NY[(flights_data_NY[month_col_flights] == max_month) & (flights_data_NY[year_col_flights] == max_year)].copy()
    Y_data[date_col_flights] = pd.to_datetime(Y_data[date_col_flights])
    # Compute ratio of daily cancellations, that is, num(cancelled flights) / num(total flights) per day for Y data
    Y_aggr = Y_data.groupby([date_col_flights, day_col_flights, year_col_flights, month_col_flights]).size().reset_index(name=num_flights)
    Y_aggr[num_cancelled] = Y_data.groupby(date_col_flights)[cancelled].sum().reset_index(drop=True)
    Y_aggr[cancelled_ratio] = Y_aggr[num_cancelled] / Y_aggr[num_flights]

    # perform test using the cancelled ratio columns of both months' data
    X = X_aggr[cancelled_ratio].to_numpy()
    Y = Y_aggr[cancelled_ratio].to_numpy()

    n, m = X.size, Y.size
    X_mean = X.sum() / n
    Y_mean = Y.sum() / m

    # Calculate corrected sample variances of both Feb(X) and March(Y) samples
    X_var = math.sqrt(((X - X_mean) ** 2).sum() / (n - 1))
    Y_var = math.sqrt(((Y - Y_mean) ** 2).sum() / (m - 1))
    # sample pool standard deviation = sqrt((sample_var_X/n) + (sample_var_Y/m))
    sample_pool_std_dev = math.sqrt((X_var / n) + (Y_var / m))

    # T-Statistic for 2 sample test is given by T = diff in sample mean / sample pool standard deviation
    T = round((X_mean - Y_mean) / sample_pool_std_dev, 3)

    # Accept/Reject the null hypothesis based on Z and the critical value. Assume alpha = 0.05
    # Degrees of freedom for unpaired test = (n-1) + (m-1)
    df = n + m - 2
    alpha = 0.05
    # t_df_alpha_by_2 = 2.002 for n = 28, m = 31, alpha = 0.05
    critical_value = round(t.ppf(1 - (alpha / 2), df=df), 3)

    print("{0} Exploratory X Dataset Inference 3: One-Tailed Unpaired T-Test {0}".format(20*"-"))
    print("To compare means of daily flight cancellation ratio in NY between months of least and highest daily covid averages")
    print("Sample size of X = {0}, Sample size of Y = {1}, Sample pool std deviation = {2:.3f}".format(n, m, sample_pool_std_dev))
    # Condition for one-tailed T-test: reject H0 if T < -(critical_value)
    if T < -critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the mean of daily flight cancellation ratio in a month with no covid cases "
        "is greater than or equal to that in a month with the highest number of covid cases, as T-statistic = {0} is less than threshold {1}"
        .format(T, -critical_value))
        # P-value for one-tailed Test = CDF(T_value)
        p_value = t.cdf(T, df=df)
        if p_value < alpha:
            print("Since the p-value {0} <<< significance level alpha {1}, we can say that we can reject the null hypothesis with a great degree of confidence"
            .format(p_value, alpha))
    else:
        print("Failed to reject Null Hypothesis: We accept the alternate hypothesis that the mean of daily flight cancellation ratio in a month with no covid cases "
        "is lesser that in a month with the highest number of covid cases, as T-statistic = {0} is not less than threshold {1}"
        .format(T, -critical_value))