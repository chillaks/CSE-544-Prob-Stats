import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Initialize subplot for plotting all line-graphs within a single plot
fig = plt.figure(figsize=(20,8))

linewidth_map = {0.0001: 1, 0.0005: 1, 0.001: 1, 0.005: 3, 0.05: 2}
color_map = {0.0001: 'forestgreen', 0.0005: 'magenta', 0.001: 'chocolate', 0.005: 'black', 0.05: 'red'}

# K(u) is the pdf of the standard 􏰇Normal at point u = 􏰌􏰍􏰌􏰎(x - x_i) / h. Hence, we replace μ = 0 and σ² = 1 in the pdf equation 
# for normal distribution 
def standard_normal_kernel_function(u):
		return (1 / math.sqrt(2 * np.pi)) * math.exp(-0.5 * (u ** 2))

# Returns KDE estimate of the PDF at point x
def normal_kde(x, h, D):
	n = len(D)
	kernel_func_sum = 0

	# Read x_i as x subscript i
	for x_i in D:
		kernel_func_sum += standard_normal_kernel_function((x - x_i) / h)
	kernel_func_sum = kernel_func_sum / (n * h)
	return kernel_func_sum

def true_normal_pdf(x, mean, std_dev):
	return norm.pdf(x, loc = mean, scale = std_dev)
    
# Given, true distribution X ~ Normal(0.5, 0.01)
true_normal_mean = 0.5
true_normal_variance = 0.01
true_normal_std_dev = math.sqrt(true_normal_variance)

# Read the second column of the csv, which contains the data points x_i in the range of i ∈ [0, 100)
D = np.genfromtxt('./datasets/a3_q7.csv', delimiter=',', skip_header=1, usecols=1)
# points at which the pdf is to be estimated, range of values from 0 to 1 with a step of 0.01
x_points = np.arange(0, 1.01, 0.01)
# smoothing bandwidth h
h_list = [0.0001, 0.0005, 0.001, 0.005, 0.05]

# Generate true PDF for all x in the range [0,1]
y_true_pdf = true_normal_pdf(x_points, true_normal_mean, true_normal_std_dev)
# Calculate mean and variance of the true PDF for x = {0, 0.01, 0.02, ..., 1}. We use the result of A3_Q4a) to compute both mean and variance
true_pdf_mean = round(y_true_pdf.sum() / len(y_true_pdf), 3)
true_pdf_variance = round(((x_points - true_pdf_mean) ** 2).sum() / len(x_points), 3)

sub_plot_num = 1
# Generate KDE estimates of PDF for all x in the range [0,1] for the given sample dataset, for different values of smoothing bandwidth h
for h in h_list:
	y_kde = np.array([normal_kde(x, h, D) for x in x_points])
	ax = fig.add_subplot(3, 2, sub_plot_num)
	ax.plot(x_points, y_kde, lw=linewidth_map.get(h), label='kde', color=color_map.get(h))
	ax.plot(x_points, y_true_pdf, lw=2, label='True Normal', color='b')
	ax.set_title('Kernel Density Estimation for Normal Distribution at h=' + str(h))
	ax.set_xlabel('x')
	ax.set_ylabel('KDE / f(x)')
	ax.grid(True)
	ax.legend(loc="upper left")
	sub_plot_num +=1

	# Calculate estimate mean and variance for the computed KDE estimate of the PDF. We use the result of A3_Q4a) to compute both mean and variance
	sample_kde_mean = round(y_kde.sum() / len(y_kde), 3)
	sample_kde_variance = round(((x_points - sample_kde_mean) ** 2).sum() / len(x_points), 3)
	
	# The deviation of KDE estimate mean/variance as a percentage difference with respect to mean/variance of true PDF
	mean_diff_perc = ((sample_kde_mean - true_pdf_mean) / true_pdf_mean) * 100
	variance_diff_perc = ((sample_kde_variance - true_pdf_variance) / true_pdf_variance) * 100
	print('For h = {0:.4f}'.format(h).rstrip('0'))
	print('KDE mean     = {0:.3f}, which has a deviation of {1:.2f}% from true PDF mean     = {2:.3f}'.format(sample_kde_mean, mean_diff_perc, true_pdf_mean))
	print('KDE variance = {0:.3f}, which has a deviation of {1:.2f}% from true PDF variance = {2:.3f}\n'.format(sample_kde_variance, variance_diff_perc, true_pdf_variance))

plt.tight_layout()
plt.show()
