import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm

# lower and upper limit (both inclusive) between which we draw integer samples. Hence, the distribution is uniform with a=low and b=high
low, high = 1, 99

def draw_ecdf_graph(x, y, sample_size, title, num_students=None):
	# plot eCDF
	plt.figure(title + ' eCDF', figsize=(20,8))
	plt.step(x + [100], y + [1], where='post', lw = 2, label='eCDF')
	plt.scatter(x[1:], [0]*(len(x[1:])), color='red', marker='x', s=50, label='samples')

	# plot true CDF
	uniform_distribution = uniform(loc=low, scale=high-low)
	x_true = np.linspace(uniform_distribution.ppf(0), uniform_distribution.ppf(1), sample_size)
	cdf = uniform_distribution.cdf(x_true)
	plt.plot(x_true, cdf, 'y-.', lw=2, label='True CDF')

	# format/update graph ticks, labels legend and title
	if num_students == None:
		plt.title(title + ' eCDF with %d samples. Sample mean = %.2f' % (len(x[1:]), np.mean(x)), fontsize=18)
	else:
		plt.title(title + ' eCDF with %d samples and number of students m=%d' % (sample_size, num_students), fontsize=18)

	plt.xticks(np.arange(0, high+10, 10))
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.xlabel('Sample x')
	plt.ylabel('CDF / Pr(X <= x)')
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()

# returns the sorted list of samples and the CDF of each RV drawn from the sample
def get_cdf_3a(sample_list):
	num_samples = len(sample_list)

	# sort the samples in asc order
	sample_list.sort()

	# initialize the lists that will contain the values to be represented on both X and Y axes. Since the graph will start at 0, we initialize
	# the first element of both to 0. X-axis: Each sample x generated from the uniform distribution function, Y-axis: CDF of each sample x
	x_sample = [0]
	y_cdf = [0]

	# Pr(X<=x), that is, probablity of a random sample selected from the uniform distribution taking a value <= each sample data in the sample list
	cumulative_pr = 0
	for sample_data in sample_list:
		cumulative_pr += 1 / num_samples
		x_sample.append(sample_data)
		y_cdf.append(cumulative_pr)

	return x_sample, y_cdf

# 3a) taking arbitrary list of samples
sample_list = [2, 56, 24, 67, 8, 24, 72, 24, 10, 45]
x_sample, y_cdf = get_cdf_3a(sample_list)
draw_ecdf_graph(x_sample, y_cdf, sample_size=len(sample_list), title='3a)')

# 3b)
n = [10, 100, 1000]
for size in n:
	sample_list = np.random.randint(low = low, high = high + 1, size = size)
	x_sample, y_cdf = get_cdf_3a(sample_list)
	draw_ecdf_graph(x_sample, y_cdf, sample_size=size, title='3b)')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Return increasing list of all possible values for RV X ~ Uniform(1, 99) and the avg eCDF for every value of X ~ Uniform(1, 99) sampled across all students
def get_avg_cdf_3c(student_sample_list):

	# number of students = number of rows in 2D matrix
	num_students = student_sample_list.shape[0]

	# Initialize a 2D array that represents the eCDF of all possible samples which can be generated for each student.
	# Any cell student_samples_cdf[i,j] represents the probability of student (i+1) selecting a RV which is less than or equal to the value j.
	# Thus, number of rows = number of students and number of columns = all possible integer values of random variable X ~ Uniform(1, 99). 
	# Eg: student_samples_cdf[1, 23] = Pr(Finding an integer x in the range of 1 <= x <= 23 in student 2's list of sample data)
	student_samples_cdf = np.zeros((num_students, high+1))

	for student_i in range(num_students):
		# get sorted list of samples and eCDF for student i
		sample_i_list, cdf_i_list = get_cdf_3a(student_sample_list[student_i])

		sample_index = 0
		for sample in sample_i_list:
			student_samples_cdf[student_i][sample] = cdf_i_list[sample_index]
			sample_index += 1

		# Backfill CDF for values which are not present in the student's sample list. If V1 and V2 (V1 < V2) are 2 successive values in student i's sorted sample list
		# for which we calculated CDF C1 and C2 respectively, all integers in the range of (V1, V2), which are not part of student i's sample list will have CDF of C1.
		prev_cdf = 0
		for sample_rv in range(1, high+1):
			if student_samples_cdf[student_i][sample_rv] != 0:
				prev_cdf = student_samples_cdf[student_i][sample_rv]
			else:
				student_samples_cdf[student_i][sample_rv] = prev_cdf

	# Calculate avg eCDF for each x across all students, i.e, average eCDF of eCDFs for each student for every value of X ~ Uniform(1,99) (axis = 0)
	avg_emp_cdf = student_samples_cdf.mean(axis = 0)
	return [i for i in range(0, high+1)], list(avg_emp_cdf)

# 3c)
student_sample_list = np.random.randint(low = low, high = high + 1, size = (5,5))
x_sample, y_cdf = get_avg_cdf_3c(student_sample_list)
draw_ecdf_graph(x_sample, y_cdf, sample_size=student_sample_list.shape[1], num_students=student_sample_list.shape[0], title='3c)')

# 3d)
m = [10, 100, 1000]
for num_students in m:
	student_sample_list = np.random.randint(low = low, high = high + 1, size = (num_students, 10))
	x_sample, y_cdf = get_avg_cdf_3c(student_sample_list)
	draw_ecdf_graph(x_sample, y_cdf, sample_size=10, num_students=num_students, title='3d)')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot Normal-based / DKW based CI lines around the eCDF, i.e, F_Cap, which is the empirical CDF for the given list of samples x
def draw_ci_lines_3ef(x, y_f_cap, alpha, title, normal_based, dkw_based):
	# Plot the eCDF for the given sample list x
	plt.figure(title + ' eCDF with CI', figsize=(20,8))
	plt.step(x, y_f_cap, where='post', lw = 2, label='F_Cap')

	# calculate n excluding 0 element
	n = y_f_cap.shape[0] - 1
	# z_alpha_by_2 ~= 1.96 when alpha = 0.05
	z_alpha_by_2 = norm.ppf(1 - (alpha / 2))

	if normal_based:
		# Normal-based CI lower limit = f_cap - (z_alpha_by_2 * sqrt(f_cap * (1-f_cap) / n))
		y_normal_ci_ll = y_f_cap - (z_alpha_by_2 * ((y_f_cap * (1 - y_f_cap) / float(n)).round(5) ** 0.5))
		# Normal-based CI upper limit = f_cap + (z_alpha_by_2 * sqrt(f_cap * (1-f_cap) / n))
		y_normal_ci_ul = y_f_cap + (z_alpha_by_2 * ((y_f_cap * (1 - y_f_cap) / float(n)).round(5) ** 0.5))

		# Plot the Normal based CI lines around F_Cap
		plt.step(x, y_normal_ci_ll, where='post', label = 'lower normal CI')
		plt.step(x, y_normal_ci_ul, where='post', label = 'upper normal CI')

	if dkw_based:
		# DKW-based CI lower limit = f_cap - (sqrt(ln(2 / alpha) / 2n)
		y_dkw_ci_ll = y_f_cap - math.sqrt(math.log(2 / alpha) / float(2*n))
		# DKW-based CI upper limit = f_cap + (sqrt(ln(2 / alpha) / 2n)
		y_dkw_ci_ul = y_f_cap + math.sqrt(math.log(2 / alpha) / float(2*n))

		# Plot the Normal based CI lines around F_Cap
		plt.step(x, y_dkw_ci_ll, where='post', label = 'lower DKW CI')
		plt.step(x, y_dkw_ci_ul, where='post', label = 'upper DKW CI')

	plt.title(title + ' eCDF and CI Lines for %d samples. Sample mean = %.2f' % (n, x.mean()), fontsize=18)
	plt.xlabel('Sample x')
	plt.ylabel('CDF / Pr(X <= x)')
	plt.grid(True)
	plt.xlim([0,2])
	plt.legend(loc='upper left')
	plt.show()

# 3e, 3f)
ci_data_samples = np.genfromtxt('./datasets/a3_q3.csv')
x_sample, y_cdf = get_cdf_3a(ci_data_samples)
x_sample, y_cdf = np.array(x_sample), np.array(y_cdf)

# We need to take 95% CI. Significance level, alpha, is given by
alpha = 1 - 0.95

# Normal based CI
draw_ci_lines_3ef(x_sample, y_cdf, alpha, '3e)', True, False)
# DKW-based CI
draw_ci_lines_3ef(x_sample, y_cdf, alpha, '3f_a)', False, True)
# Normal and DKW based CI
draw_ci_lines_3ef(x_sample, y_cdf, alpha, '3f_b)', True, True)

