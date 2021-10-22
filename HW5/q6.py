import math
import numpy as np
from scipy.stats import norm, t
#from statsmodels.stats import weightstats as stests

# Given X ~ Normal(1.5, 1), Y ~ Normal(1, 1) 
# 6 a) Z and T tests for 20-sized sample data
print('-------------------------------------6 a)--------------------------------')
X1 = np.genfromtxt('./datasets/q6_X1.csv', autostrip=True, skip_header=1)
Y1 = np.genfromtxt('./datasets/q6_Y1.csv', autostrip=True, skip_header=1)
#stests.ztest(x1=X1, x2=Y1, value=0, alternative='two-sided')

# (i) 2-Sample, 2-Tailed Z-Test
X_true_std_dev = 1
Y_true_std_dev = 1

# n = size(X1), m = size(Y1)
n, m = len(X1), len(Y1)

# Under null hypothesis, means of X and Y are same. Therefore the hypothesized difference between the population means will be 0
delta = 0
# Calculate sample means X_bar and Y_bar
X1_mean = X1.sum() / n
Y1_mean = Y1.sum() / m
# sample pool standard deviation = sqrt((sigma_X^2/n) + (sigma_Y^2/m))
sample_pool_std_dev_Z = math.sqrt((X_true_std_dev ** 2 / n) + (Y_true_std_dev ** 2 / m))
print("--> 2-Sample, 2-Tailed Z-Test with sample size 20:")
print("X1 sample mean = {0:.4f}, Y1 sample mean = {1:.4f}, sample pool standard deviation = {2:.4f}".format(X1_mean, Y1_mean, sample_pool_std_dev_Z))

# Z-score for 2 sample test is given by Z = (diff in sample mean - delta) / sample pool standard deviation
z_stat = ((X1_mean - Y1_mean) - delta) / sample_pool_std_dev_Z
# P-score for Z-test = 2 * (1 - phi(z_stat))
p_value_z_20 = 2 * (1 - norm.cdf(abs(z_stat)))
print("Observed Z = {0:.4f}, p-value = {1:.4f}".format(z_stat, p_value_z_20))

# Reject/Accept Null Hypothesis based on calculated Z-statistic and given threshold/critical value
critical_value_Z = 1.962
if abs(z_stat) > critical_value_Z:
    print("Rejected Null Hypothesis: We reject the hypothesis that X and Y have the same mean value, as the Z Statistic = {0:.4f} exceeds threshold {1:.4f}. We therefore accept the alternate hypothesis that they have different means\n".format(z_stat, critical_value_Z))
else:
    print("Failed to reject Null Hypothesis: We accept the hypothesis that X and Y have the same mean value, as the Z Statistic = {0:.4f} does not exceed threshold {1:.4f}\n".format(z_stat, critical_value_Z))

# (ii) Unpaired T-test
# Calculate unbiased sample variances of samples X1 and Y1. ddof=1 implies we use (sample_size - 1) in denominator
X1_var = X1.var(ddof=1)
Y1_var = Y1.var(ddof=1)
# sample pool standard deviation = sqrt((sample_var_X1/n) + (sample_var_Y1/m))
sample_pool_std_dev_T = math.sqrt((X1_var / n) + (Y1_var / m))
print("--> Unpaired T-test with sample size 20:")
print("X1 sample variance = {0:.4f}, Y1 sample variance = {1:.4f}, sample pool standard deviation = {2:.4f}".format(X1_var, Y1_var, sample_pool_std_dev_T))

# T-Statistic for 2 sample test is given by T = diff in sample mean / sample pool standard deviation
t_stat = (X1_mean - Y1_mean) / sample_pool_std_dev_T
# Degrees of freedom for unpaired test = (n-1) + (m-1)
df_t = n + m - 2
# P-score for T-test = 2 * (1 - cdf(t_stat))
p_value_t_20 = 2 * (1 - t.cdf(t_stat, df=df_t))
print("Observed T = {0:.4f}, p-value = {1:.4f}".format(t_stat, p_value_t_20))

# Reject/Accept Null Hypothesis based on calculated T-statistic and given threshold/critical value
critical_value_T = 2.086
if abs(t_stat) > critical_value_T:
    print("Rejected Null Hypothesis: We reject the hypothesis that X and Y have the same mean value, as the T Statistic = {0:.4f} exceeds threshold {1:.4f}. We therefore accept the alternate hypothesis that they have different means\n".format(t_stat, critical_value_T))
else:
    print("Failed to reject Null Hypothesis: We accept the hypothesis that X and Y have the same mean value, as the T Statistic = {0:.4f} does not exceed threshold {1:.4f}\n".format(t_stat, critical_value_T))

# 6 b) Z and T tests for 1000-sized sample data
print('-------------------------------------6 b)--------------------------------')
X2 = np.genfromtxt('./datasets/q6_X2.csv', autostrip=True, skip_header=1)
Y2 = np.genfromtxt('./datasets/q6_Y2.csv', autostrip=True, skip_header=1)
#stests.ztest(x1=X2, x2=Y2, value=0, alternative='two-sided')

# (i) 2-Sample, 2-Tailed Z-Test
# n = size(X2), m = size(Y2)
n, m = len(X2), len(Y2)

# Under null hypothesis, means of X and Y are same. Therefore the hypothesized difference between the population means will be 0
delta = 0
# Calculate sample means X_bar and Y_bar
X2_mean = X2.sum() / n
Y2_mean = Y2.sum() / m
# sample pool standard deviation = sqrt((sigma_X^2/n) + (sigma_X^2/m))
sample_pool_std_dev_Z = math.sqrt((X_true_std_dev ** 2 / n) + (Y_true_std_dev ** 2 / m))
print("--> 2-Sample, 2-Tailed Z-Test with sample size 1000:")
print("X2 sample mean = {0:.4f}, Y2 sample mean = {1:.4f}, sample pool standard deviation = {2:.4f}".format(X2_mean, Y2_mean, sample_pool_std_dev_Z))

# Z-score for 2 sample test is given by Z = (diff in sample mean - delta) / sample pool standard deviation
z_stat = ((X2_mean - Y2_mean) - delta) / sample_pool_std_dev_Z
# P-score for Z-test = 2 * (1 - phi(z_stat))
p_value_z = 2 * (1 - norm.cdf(abs(z_stat)))
print("Observed Z = {0:.4f}, p-value = {1:.4f}".format(z_stat, p_value_z))

# Reject/Accept Null Hypothesis based on calculated Z-statistic and given threshold/critical value
critical_value_Z = 1.962
if abs(z_stat) > critical_value_Z:
    print("Rejected Null Hypothesis: We reject the hypothesis that X and Y have the same mean value, as the Z Statistic = {0:.4f} exceeds threshold {1:.4f}. We therefore accept the alternate hypothesis that they have different means\n".format(z_stat, critical_value_Z))
else:
    print("Failed to reject Null Hypothesis: We accept the hypothesis that X and Y have the same mean value, as the Z Statistic = {0:.4f} does not exceed threshold {1:.4f}\n".format(z_stat, critical_value_Z))

# (ii) Unpaired T-test
# Calculate unbiased sample variances of samples X2 and Y2. ddof=1 implies we use (sample_size - 1) in denominator
X2_var = X2.var(ddof=1)
Y2_var = Y2.var(ddof=1)
# sample pool standard deviation = sqrt((sample_var_X2/n) + (sample_var_Y2/m))
sample_pool_std_dev_T = math.sqrt((X2_var / n) + (Y2_var / m))
print("--> Unpaired T-test with sample size 1000:")
print("X2 sample variance = {0:.4f}, Y2 sample variance = {1:.4f}, sample pool standard deviation = {2:.4f}".format(X2_var, Y2_var, sample_pool_std_dev_T))

# T-Statistic for 2 sample test is given by T = diff in sample mean / sample pool standard deviation
t_stat = (X2_mean - Y2_mean) / sample_pool_std_dev_T
# Degrees of freedom for unpaired test = (n-1) + (m-1)
df_t = n + m - 2
# P-score for T-test = 2 * (1 - cdf(t_stat))
p_value_t = 2 * (1 - t.cdf(t_stat, df=df_t))
print("Observed T = {0:.4f}, p-value = {1:.4f}".format(t_stat, p_value_t))

# Reject/Accept Null Hypothesis based on calculated T-statistic and given threshold/critical value
critical_value_T = 1.962
if abs(t_stat) > critical_value_T:
    print("Rejected Null Hypothesis: We reject the hypothesis that X and Y have the same mean value, as the T Statistic = {0:.4f} exceeds threshold {1:.4f}. We therefore accept the alternate hypothesis that they have different means\n".format(t_stat, critical_value_T))
else:
    print("Failed to reject Null Hypothesis: We accept the hypothesis that X and Y have the same mean value, as the T Statistic = {0:.4f} does not exceed threshold {1:.4f}\n".format(t_stat, critical_value_T))

print("--------------------------------------------------------------------------")
print("""Observations on p-value in parts a and b:
1. We got a True Positive result in part b), as we were successfully able to reject the Null Hypothesis, that is, determine that the means of X and Y are not same, given we already have the ground truth 
that mean of X is 1.5 and mean of Y is 1. We can therefore claim that both Z and T tests work well with larger data samples.
2. There is not much of a significant advantage of using Z-Test on small samples, as both tests on the 20-sample data failed to reject the Null Hypothesis even though we know that the means of X and Y 
are not same, thereby resulting in a False Negative. However, the p-value is much closer to alpha=0.05 in the case of Z-Test({0:.4f}) than in the T-Test({1:.4f})
3. We can see that the p-value is significantly lesser than alpha = 0.05 in part b) with 1000 samples for both Z and T Tests. Therefore, we can support our rejection of the Null hypothesis with extremely high confidence,
as 0.0000 <<<< 0.05""".format(p_value_z_20, p_value_t_20))
