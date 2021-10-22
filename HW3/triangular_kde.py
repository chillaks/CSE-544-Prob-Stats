import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import triang

# Triangular distribution has 3 points left, right, mode. Always, left <= mode <= right. mode represents the 
# peak value/height of the triangle. For the given range of -1 <= u <= 1, u peaks at x=0, hence we have
left, right, mode = -1, 1, 0

# Calculate the parameters needed by scipy's triang method to generate true PDF values for a triangular distribution
c = (mode - left) / (right - left)
loc = left
scale = right - left
triangular_distribution = triang(c=c, loc=loc, scale=scale)

# Initialize subplot for plotting all line-graphs within a single plot
fig = plt.figure(figsize=(20,8))

linewidth_map = {0.0001: 1, 0.0005: 1, 0.001: 1, 0.005: 2, 0.05: 2}
color_map = {0.0001: 'forestgreen', 0.0005: 'magenta', 0.001: 'chocolate', 0.005: 'black', 0.05: 'red'}

# Given K(u) is 1 - abs(u) when in range [-1, 1] and 0 otherwise
def triangular_kernel_function(u):
	absolute_u = round(abs(u), 4)
	return (1 - absolute_u) if absolute_u <= right else 0

# Returns KDE estimate of the PDF at point x
def triangular_kde(x, h, D):
	n = len(D)
	kernel_func_sum = 0

	# Read x_i as x subscript i
	for x_i in D:
		kernel_func_sum += triangular_kernel_function((x - x_i) / h)
	kernel_func_sum = kernel_func_sum / (n * h)
	return kernel_func_sum

def true_triangular_pdf(x):
	return triangular_distribution.pdf(x)

# Read the second column of the csv, which contains the data points x_i in the range of i ∈ [0, 100)
D = np.genfromtxt('./datasets/a3_q7.csv', delimiter=',', skip_header=1, usecols=1)
# points at which the pdf is to be estimated, range of values from -1 to 1
x_points = np.linspace(triangular_distribution.ppf(0), triangular_distribution.ppf(1), len(D))
# smoothing bandwidth h
h_list = [0.0001, 0.0005, 0.001, 0.005, 0.05]

# Generate true PDF for all x in the range [-1, 1]
y_true_pdf = true_triangular_pdf(x_points)
# Calculate mean and variance of the true PDF for x = {0, 0.01, 0.02, ..., 1}. We use the result of A3_Q4a) to compute both mean and variance
true_pdf_mean = round(y_true_pdf.sum() / len(y_true_pdf), 3)
true_pdf_variance = round(((x_points - true_pdf_mean) ** 2).sum() / len(x_points), 3)

sub_plot_num = 1
# Generate KDE estimates of PDF for all x in the range [-1,1] for the given sample dataset, for different values of smoothing bandwidth h
for h in h_list:
	y_kde = np.array([triangular_kde(x, h, D) for x in x_points])
	ax = fig.add_subplot(3, 2, sub_plot_num)
	ax.plot(x_points, y_kde, lw=linewidth_map.get(h), label='kde', color=color_map.get(h))
	ax.plot(x_points, y_true_pdf, lw=2, label='True Triangular', color='b')
	ax.set_title('Kernel Density Estimation for Triangular Distribution at h=' + str(h))
	ax.set_xlabel('x')
	ax.set_ylabel('KDE / f(x)')
	ax.grid(True)
	ax.legend(loc="upper left")
	sub_plot_num += 1

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
