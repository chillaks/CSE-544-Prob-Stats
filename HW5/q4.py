import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Uncomment below to view more rows
# pd.set_option('display.max_rows', 50)

print('----------------------------------------- PART a) --------------------------------------')
# Read the mixed stroke-glucose levels dataset
sample_df = pd.read_csv('./datasets/data_q4_1.csv')
sample_df
# Split the mixed dataset into 2 sets (Stroke and No Stroke glucose levels)
stroke_glucose_samples = sample_df.loc[sample_df['stroke'] == 1]['avg_glucose_level'].to_numpy()
no_stroke_glucose_samples = sample_df.loc[sample_df['stroke'] == 0]['avg_glucose_level'].to_numpy()

# Number of samples of glucose levels of all stroke and no-stroke patients in given dataset
s_size = len(stroke_glucose_samples)
n_size = len(no_stroke_glucose_samples)
# Mean glucose levels of all stroke and no-stroke patients
s_mean = stroke_glucose_samples.sum() / s_size
n_mean = no_stroke_glucose_samples.sum() / n_size

# Calculate T_obs, which is the difference in glucose level means of the original samples
T_obs = abs(s_mean - n_mean)
print("T_obs =", T_obs)

# No of permutation
n = [200, 1000]
# Join the stroke and no-stroke glucose level samples to a single list
combined_df = np.concatenate([stroke_glucose_samples, no_stroke_glucose_samples])

for num_perm in n:
    print("Number of permutations n = {0}:".format(num_perm))
    # Define counter to track number of times we observe the absolute difference in sample means of the permutations of original samples to be greater than T_obs  
    indicator_var = 0
    j = 0
    for i in range(num_perm):
        # Shuffle combined dataset to generate random permutations of initial samples, that is, intermixing the samples keeping the orginal sample size the same
        np.random.shuffle(combined_df)
        s_bar_samples = combined_df[:s_size]
        n_bar_samples = combined_df[s_size:]

        # Calculate T_i, which is difference in means of the mixed samples after permutation
        s_bar_mean = s_bar_samples.sum() / s_size
        n_bar_mean = n_bar_samples.sum() / n_size
        T_i = abs(s_bar_mean - n_bar_mean)

        # Increment counter if T_i > T_obs
        indicator_var = indicator_var + 1 if T_i > T_obs else indicator_var
    print("Observed T_i > T_obs {0} times".format(indicator_var))

    # Perform Permutation Test
    # P-value is the probability of seeing an absolute difference in mean with a more extreme value than T_obs, that is, Pr(T_i > T_obs) under null hypothesis
    p_value = indicator_var / num_perm
    p_value_threshold = 0.05
    print("Calculated p-value =",p_value)

    if p_value <= p_value_threshold:
        print("Rejected Null Hypothesis: We reject the hypothesis that people getting stroke tend to have the same glucose level as people who do not get stroke, as p-value={0} does not exceed threshold {1}".format(p_value, p_value_threshold))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that people getting stroke tend to have the same glucose level as people who do not get stroke, as p-value={0} exceeds threshold {1}".format(p_value, p_value_threshold))
    print()


print('----------------------------------------- PART b) --------------------------------------')
# Read the mixed male-female stroke age dataset
sample_df = pd.read_csv('./datasets/data_q4_2.csv')
# Split the mixed dataset into 2 sets (Male and Female)
female_age_samples = sample_df.loc[sample_df['gender'] == 'Female']['age'].to_numpy()
male_age_samples = sample_df.loc[sample_df['gender'] == 'Male']['age'].to_numpy()

# Number of samples of male and female ages in given dataset
f_size = len(female_age_samples)
m_size = len(male_age_samples)
# Mean age of all males and females in the dataset
f_mean = female_age_samples.sum() / f_size
m_mean = male_age_samples.sum() / m_size

# Calculate T_obs, which is the difference in mean ages of the original samples
T_obs = abs(f_mean - m_mean)
print("T_obs =", T_obs)

# No of permutation
n = 1000
# Join the male and female age samples to a single list
combined_df = np.concatenate([female_age_samples, male_age_samples])

# Define counter to track number of times we observe the absolute difference in sample means of the permutations of original samples to be greater than T_obs  
indicator_var = 0
j = 0
for i in range(n):
    # Shuffle combined dataset to generate random permutations of initial samples, that is, intermixing the samples keeping the orginal sample size the same
    np.random.shuffle(combined_df)
    f_bar_samples = combined_df[:f_size]
    m_bar_samples = combined_df[f_size:]

    # Calculate T_i, which is difference in means of the mixed samples after permutation
    f_bar_mean = f_bar_samples.sum() / f_size
    m_bar_mean = m_bar_samples.sum() / m_size
    T_i = abs(f_bar_mean - m_bar_mean)

    # Increment counter if T_i > T_obs
    indicator_var = indicator_var + 1 if T_i > T_obs else indicator_var
print("Observed T_i > T_obs {0} times out of {1} permutations".format(indicator_var, n))

# Perform Permutation Test
# P-value is the probability of seeing an absolute difference in mean with a more extreme value than T_obs, that is, Pr(T_i > T_obs) under null hypothesis
p_value = indicator_var / n
p_value_threshold = 0.05
print("Calculated p-value =",p_value)

if p_value <= p_value_threshold:
    print("Rejected Null Hypothesis: We reject the hypothesis that female patients get a stroke at the same age as male patients, as p-value={0} does not exceed threshold {1}".format(p_value, p_value_threshold))
else:
    print("Failed to reject Null Hypothesis: We accept the hypothesis that female patients get a stroke at the same age as male patients, as p-value={0} exceeds threshold {1}".format(p_value, p_value_threshold))
print()


print('----------------------------------------- PART c) --------------------------------------')
# Read the mixed male-female stroke dataset
sample_df = pd.read_csv('./datasets/data_q4_2.csv')

# Calculate CDF at every 'age' point on both Male and Female datasets
def get_cdf_list(df):
    cdf_list = []
    num_samples = df.shape[0]

    cumulative_pr = 0
    for index, row in df.iterrows():
        cumulative_pr += 1 / num_samples
        cdf_list.append(cumulative_pr)
    
    return cdf_list

# Split the mixed dataset into 2 sets (Male and Female). Sort them in ascending order of age
female_df = sample_df.loc[sample_df['gender'] == 'Female'].sort_values('age').reset_index(drop=True)
male_df = sample_df.loc[sample_df['gender'] == 'Male'].sort_values('age').reset_index(drop=True)
# Add a new column denoting the CDF at each point in the male and female datasets
female_df['female_eCDF'] = get_cdf_list(female_df)
male_df['male_eCDF'] = get_cdf_list(male_df)

# Find distinct datapoints and their corresponding CDFs at each of the points in both male and female sets
male_distinct_df = male_df.drop_duplicates(subset='age', keep="last").reset_index(drop=True)
female_distinct_df = female_df.drop_duplicates(subset='age', keep="last").reset_index(drop=True)

if male_df.shape[0] > female_df.shape[0]:
    x_age = female_distinct_df['age'].to_numpy()
else:
    x_age = male_distinct_df['age'].to_numpy()

KS_Test_cols = ['x', 'F_cap_male_left', 'F_cap_male_right', 'F_cap_female_left', 'F_cap_female_right', 'left_diff_abs', 'right_diff_abs']
row_list = []

for x in x_age:
    # Find CDF to the left of point x in the sorted male dataset
    male_left_cdf = male_distinct_df.loc[male_distinct_df['age'] < x, 'male_eCDF']
    F_cap_male_left = 0.0 if male_left_cdf.empty else male_left_cdf.max()
    # Find CDF to the right of point x in the sorted male dataset
    male_right_cdf = male_distinct_df.loc[male_distinct_df['age'] >= x, 'male_eCDF']
    F_cap_male_right = 0.0 if male_right_cdf.empty else male_right_cdf.min()
    # Find CDF to the left of point x in the sorted female dataset
    female_left_cdf = female_distinct_df.loc[female_distinct_df['age'] < x, 'female_eCDF']
    F_cap_female_left = 0.0 if female_left_cdf.empty else female_left_cdf.max()
    # Find CDF to the left of point x in the sorted female dataset
    female_right_cdf = female_distinct_df.loc[female_distinct_df['age'] >= x, 'female_eCDF']
    F_cap_female_right = 0.0 if female_right_cdf.empty else female_right_cdf.min()
    # Find absolute difference between left CDFs of male and female datasets
    left_diff_abs = round(abs(F_cap_male_left - F_cap_female_left), 4)
    # Find absolute difference between right CDFs of male and female datasets
    right_diff_abs = round(abs(F_cap_male_right - F_cap_female_right), 4)
    
    # Build each row to be appended to the KS Test Table 
    row = [x, F_cap_male_left, F_cap_male_right, F_cap_female_left, F_cap_female_right, left_diff_abs, right_diff_abs]
    row_dict = dict(zip(KS_Test_cols, row))
    row_list.append(row_dict)

# Build KS Test Table (represented as a dataframe)
KS_Test_df = pd.DataFrame(row_list, columns=KS_Test_cols)
print(KS_Test_df)

# Calculate KS statistic
x_points = []
d_right = KS_Test_df.iloc[KS_Test_df['right_diff_abs'].idxmax(axis=1)][['x', 'right_diff_abs']]
d_left = KS_Test_df.iloc[KS_Test_df['left_diff_abs'].idxmax(axis=1)][['x', 'left_diff_abs']]
if d_right['right_diff_abs'] == d_left['left_diff_abs']:
    print("KS Statistic is {0} at age points {1} and {2}".format(d_right['right_diff_abs'], d_left['x'], d_right['x']))
    x_points.append(d_right['x'])
    x_points.append(d_left['x'])
elif d_right['right_diff_abs'] > d_left['left_diff_abs']:
    print("KS Statistic is {0} at {1}".format(d_right['right_diff_abs'], d_right['x']))
    x_points.append(d_right['x'])
else:
    print("KS Statistic is {0} at {1}".format(d_left['left_diff_abs'], d_left['x']))
    x_points.append(d_left['x'])

# Reject/Accept Null Hypothesis based on calculated KS Statistic d and given threshold=0.05
d = max(d_right['right_diff_abs'], d_left['left_diff_abs'])
critical_value = 0.05

if d > critical_value:
    print("Rejected Null Hypothesis: We reject the hypothesis that female patients get a stroke at the same age as male patients, as KS Statistic d={0} exceeds threshold {1}".format(d, critical_value))
else:
    print("Failed to reject Null Hypothesis: We accept the hypothesis that female patients get a stroke at the same age as male patients, as KS Statistic d={0} does not exceed threshold {1}".format(d, critical_value))

# Plot KS Test eCDF
plt.figure('KS Test eCDF', figsize=(20,8))
plt.step(male_df['age'].to_numpy(), male_df['male_eCDF'], where='post', lw = 1.5, label='Male eCDF')
plt.step(female_df['age'].to_numpy(), female_df['female_eCDF'], where='post', lw = 1.5, label='Female eCDF')
for x in x_points:
    plt.axvline(x, linestyle="dashed", lw=1)

plt.xlabel('Age')
plt.ylabel('eCDF')
plt.legend(loc='upper left')
plt.grid()
plt.show()
