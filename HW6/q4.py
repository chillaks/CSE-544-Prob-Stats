import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('----------------------------------------- PART a) --------------------------------------')
# Read population dataset, containing the total population and population of 65+ year olds (in millions) per year from 1980-2019
population_data = pd.read_csv('./datasets/q4.csv', names=['Year', 'Population', 'Population_65+'], header=0)
population_total = population_data['Population'].to_numpy()
population_65p = population_data['Population_65+'].to_numpy()
year = population_data['Year'].to_numpy()

# 1) Perform SLR of Total Population vs Year. X = Year, Y = Total Population
# β1_cap = sigma((X_i - X_bar) * (Y_i - Y_bar())) / sigma((X_i - X_bar)^2), using result of A6_3a
beta_cap_1 = round(((year - year.mean()) * (population_total - population_total.mean())).sum() / ((year - year.mean()) ** 2).sum(), 4)
# β0_cap = Y_bar - β1_cap*X_bar
beta_cap_0 = round(population_total.mean() - (beta_cap_1 * year.mean()), 4)
print("For Total Population, intercept β0_cap = {0} and X coefficient β1_cap = {1}".format(beta_cap_0, beta_cap_1))
# Y_i_cap = β0_cap + β1_cap * X_i
population_total_pred = beta_cap_0 + (beta_cap_1 * year)

# Calculate SSE for Total Population
residual_cap = population_total - population_total_pred
sse_total = (residual_cap ** 2).sum()
print("SSE for Total Population dataset = {0}".format(round(sse_total, 4)))

# 2) Perform SLR of Population_65+ vs Year. X = Year, Y = Population of 65+ year olds
beta_cap_1 = round(((year - year.mean()) * (population_65p - population_65p.mean())).sum() / ((year - year.mean()) ** 2).sum(), 4)
beta_cap_0 = round(population_65p.mean() - (beta_cap_1 * year.mean()), 4)
print("For 65+ Population, intercept β0_cap = {0} and X coefficient β1_cap = {1}".format(beta_cap_0, beta_cap_1))
# Y_i_cap = β0_cap + β1_cap * X_i
population_65p_pred = beta_cap_0 + (beta_cap_1 * year)

# Calculate SSE for 65+ Population
residual_cap = population_65p - population_65p_pred
sse_65p = (residual_cap ** 2).sum()
print("SSE for 65+ Population dataset = {0}\n".format(round(sse_65p, 4)))

def plot_fit(x, y_actual, y_pred, label, sse):
	plt.figure(label + ' Population vs Year', figsize=(20,8))
	plt.plot(x, y_actual, label = "Actual " + label + " Population")
	plt.plot(x, y_pred, label = "Predicted " + label + " Population")

	plt.title('SLR Fit for ' + label + ' Population vs Year. SSE = %.4f' % (sse))
	plt.xlabel('Year')
	plt.ylabel(label + ' Population (in Millions)')
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()

# Plot the SLR Fit for Total Population
plot_fit(year, population_total, population_total_pred, "Total", sse_total)
# Plot the SLR Fit for 65+ Population
plot_fit(year, population_65p, population_65p_pred, "65+", sse_65p)

print('----------------------------------------- PART b) --------------------------------------')
# Training data using 65+ population from 1980 to 2018
last_39y_train = population_65p[:-1]
last_39_years = year[:-1]
# Training data using 65+ population from 2008 to 2018
last_11_years = year[-12:-1]
last_11y_train = population_65p[-12:-1]
last_11y_total_train = population_total[-12:-1]
last_11y_ratio_train = last_11y_train / last_11y_total_train

# Find Beta values for 1980-2018 data
beta_cap_1_39y = round(((last_39_years - last_39_years.mean()) * (last_39y_train - last_39y_train.mean())).sum() / ((last_39_years - last_39_years.mean()) ** 2).sum(), 4)
beta_cap_0_39y = round(last_39y_train.mean() - (beta_cap_1_39y * last_39_years.mean()), 4)
print("For 65+ Population from 1980-2018, intercept β0_cap = {0} and X coefficient β1_cap = {1}".format(beta_cap_0_39y, beta_cap_1_39y))
# Predict 65+ Population in 2060 using the beta values calculated above
population_65p_2060 = round(beta_cap_0_39y + (beta_cap_1_39y * 2060), 4)
print("Predicted 65+ Population in 2060 using 1980-2018 data = {0} million".format(population_65p_2060))
# Calculate SSE for 1980-2018 65+ Population data
y_pred_39y = beta_cap_0_39y + (beta_cap_1_39y * last_39_years)
residual_cap = last_39y_train - y_pred_39y
sse = (residual_cap ** 2).sum()
print("SSE for 1980-2018 65+ Population data = {0}\n".format(round(sse, 4)))


# Find Beta values for 2008-2018 data
beta_cap_1_11y = round(((last_11_years - last_11_years.mean()) * (last_11y_train - last_11y_train.mean())).sum() / ((last_11_years - last_11_years.mean()) ** 2).sum(), 4)
beta_cap_0_11y = round(last_11y_train.mean() - (beta_cap_1_11y * last_11_years.mean()), 4)
print("For 65+ Population from 2008-2018, intercept β0_cap = {0} and X coefficient β1_cap = {1}".format(beta_cap_0_11y, beta_cap_1_11y))
# Predict 65+ Population in 2060 using the beta values calculated above
population_65p_2060 = round(beta_cap_0_11y + (beta_cap_1_11y * 2060), 4)
print("Predicted 65+ Population in 2060 using 2008-2018 data = {0} million".format(population_65p_2060))
# Calculate SSE for 2008-2018 65+ Population data
y_pred_11y = beta_cap_0_11y + (beta_cap_1_11y * last_11_years)
residual_cap = last_11y_train - y_pred_11y
sse = (residual_cap ** 2).sum()
print("SSE for 2008-2018 65+ Population data = {0}\n".format(round(sse, 4)))

print('----------------------------------------- PART c) --------------------------------------')
print("Actual 65+ Population ratio in {0} = {1}. Actual Total Population = {2}, Actual 65+ Population = {3}".format(year[-1], round(population_65p[-1] / population_total[-1], 4), population_total[-1], population_65p[-1]))
# Ratio using first method
beta_1_method_1 = round(((last_11_years - last_11_years.mean()) * (last_11y_ratio_train - last_11y_ratio_train.mean())).sum() / ((last_11_years - last_11_years.mean()) ** 2).sum(), 4)
beta_0_method_1 = round(last_11y_ratio_train.mean() - (beta_1_method_1 * last_11_years.mean()), 4)
# Predict 65+ Population ratio in 2019 using the beta values calculated above
ratio_2019_m1 = round(beta_0_method_1 + (beta_1_method_1 * 2019), 4)
print("Predicted 65+ Population ratio in 2019 using method 1 = {0}".format(ratio_2019_m1))

# Ratio using second method
# Fit model to predict total population in 2019
beta_1_method_2 = round(((last_11_years - last_11_years.mean()) * (last_11y_total_train - last_11y_total_train.mean())).sum() / ((last_11_years - last_11_years.mean()) ** 2).sum(), 4)
beta_0_method_2 = round(last_11y_total_train.mean() - (beta_1_method_2 * last_11_years.mean()), 4)
# Predict Total Population in 2019 using the beta values calculated above
population_total_2019 = round(beta_0_method_2 + (beta_1_method_2 * 2019), 4)
# Predict 65+ Population in 2019 using the beta values calculated
population_65p_2019 = round(beta_cap_0_11y + (beta_cap_1_11y * 2019), 4)
print("Predicted 65+ Population ratio in 2019 using method 2 = {0}. Predicted Total population = {1} million, Predicted 65+ population = {2} million".format(round(population_65p_2019 / population_total_2019, 4), population_total_2019, population_65p_2019))
