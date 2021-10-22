import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Uncomment below to view more rows
# pd.set_option('display.max_rows', 100)

# Calculate CDF for the number of cases/deaths on each day
def get_cdf_list(num_samples):
    cdf_list = []
    cumulative_pr = 0
    for _ in range(num_samples):
        cumulative_pr += 1 / num_samples
        cdf_list.append(cumulative_pr)
    return cdf_list

def get_left_cdf(state_df, col_name, x, eCDF_col):
    if state_df[col_name].max() < x:
        return 1
    elif state_df[col_name].min() > x:
        return 0
    else:
        left_cdf = state_df.loc[state_df[col_name] < x, eCDF_col]
        F_cap_left = 0.0 if left_cdf.empty else left_cdf.max()
        return F_cap_left

def get_right_cdf(state_df, col_name, x, eCDF_col):
    if state_df[col_name].max() < x:
        return 1
    elif state_df[col_name].min() > x:
        return 0
    else:
        right_cdf = state_df.loc[state_df[col_name] >= x, eCDF_col]
        F_cap_right = 0.0 if right_cdf.empty else right_cdf.min()
        return F_cap_right

def plot_KS_2_Sample_eCDF(CT_df, CT_col_name, DC_df, DC_col_name, max_diff_x, x_label):
    if CT_df[CT_col_name].min() > DC_df[DC_col_name].max():
        x_points_CT = np.insert(CT_df[CT_col_name].to_numpy(), (0, CT_df.shape[0]), (DC_df[DC_col_name].max(), CT_df[CT_col_name].max() + 100))
        y_ecdf_CT = np.insert(CT_df['CT_eCDF'].to_numpy(), (0, CT_df.shape[0]), (0, 1))
        x_points_DC = np.insert(DC_df[DC_col_name].to_numpy(), (0, DC_df.shape[0]), (DC_df[DC_col_name].min() - 100, CT_df[CT_col_name].min()))
        y_ecdf_DC = np.insert(DC_df['DC_eCDF'].to_numpy(), (0, DC_df.shape[0]), (0, 1))
    elif DC_df[DC_col_name].min() > CT_df[CT_col_name].max():
        x_points_DC = np.insert(DC_df[DC_col_name].to_numpy(), (0, DC_df.shape[0]), (CT_df[CT_col_name].max(), DC_df[DC_col_name].max() + 100))
        y_ecdf_DC = np.insert(DC_df['DC_eCDF'].to_numpy(), (0, DC_df.shape[0]), (0, 1))
        x_points_CT = np.insert(CT_df[CT_col_name].to_numpy(), (0, CT_df.shape[0]), (CT_df[CT_col_name].min() - 100, DC_df[DC_col_name].min()))
        y_ecdf_CT = np.insert(CT_df['CT_eCDF'].to_numpy(), (0, CT_df.shape[0]), (0, 1))
    else:
        x_points_CT = CT_df[CT_col_name].to_numpy()
        y_ecdf_CT = CT_df['CT_eCDF'].to_numpy()
        x_points_DC = DC_df[DC_col_name].to_numpy()
        y_ecdf_DC = DC_df['DC_eCDF'].to_numpy()
    
    plt.figure('KS 2-Sample Test eCDF', figsize=(20,8))
    plt.step(x_points_CT, y_ecdf_CT, where='post', lw = 1.5, label='CT eCDF')
    plt.step(x_points_DC, y_ecdf_DC, where='post', lw = 1.5, label='DC eCDF')
    for x in max_diff_x:
        plt.axvline(x, linestyle="dashed", lw=1)

    plt.xlabel(x_label)
    plt.ylabel('eCDF')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def KS_2_Sample_Test(states_data, column_type):
    CT_col_name = 'CT ' + column_type
    DC_col_name = 'DC ' + column_type

    # Split the dataset per state and sort the 2 state-specific columns on which we need to perform the KS Test (#cases/#deaths)
    CT_sorted_df = states_data.loc[(states_data['Date'] >= '2020-10-01') & (states_data['Date'] <= '2020-12-31')][[CT_col_name]].sort_values(CT_col_name).reset_index(drop=True)
    DC_sorted_df = states_data.loc[(states_data['Date'] >= '2020-10-01') & (states_data['Date'] <= '2020-12-31')][[DC_col_name]].sort_values(DC_col_name).reset_index(drop=True)
    
    # Add a new column denoting the CDF at each point in the CT and DC cases/deaths columns
    CT_sorted_df['CT_eCDF'] = get_cdf_list(CT_sorted_df.shape[0])
    DC_sorted_df['DC_eCDF'] = get_cdf_list(DC_sorted_df.shape[0])

    # Find distinct datapoints and their corresponding CDFs at each of the points
    CT_distinct_df = CT_sorted_df.drop_duplicates(subset=CT_col_name, keep="last").reset_index(drop=True)
    DC_distinct_df = DC_sorted_df.drop_duplicates(subset=DC_col_name, keep="last").reset_index(drop=True)

    if CT_distinct_df.shape[0] > DC_distinct_df.shape[0]:
        x_points = DC_distinct_df[DC_col_name].to_numpy()
    else:
        x_points = CT_distinct_df[CT_col_name].to_numpy()

    KS_Test_cols = ['x', 'F_cap_CT_left', 'F_cap_CT_right', 'F_cap_DC_left', 'F_cap_DC_right', 'left_diff_abs', 'right_diff_abs']
    row_list = []
    
    for x in x_points:
        # Find CDF to the left of point x in the sorted CT dataset
        F_cap_CT_left = get_left_cdf(CT_distinct_df, CT_col_name, x, 'CT_eCDF')
        # Find CDF to the right of point x in the sorted CT dataset
        F_cap_CT_right = get_right_cdf(CT_distinct_df, CT_col_name, x, 'CT_eCDF')
        # Find CDF to the left of point x in the sorted DC dataset
        F_cap_DC_left = get_left_cdf(DC_distinct_df, DC_col_name, x, 'DC_eCDF')
        # Find CDF to the right of point x in the sorted DC dataset
        F_cap_DC_right = get_right_cdf(DC_distinct_df, DC_col_name, x, 'DC_eCDF')
        # Find absolute difference between left CDFs of CT and DC datasets
        left_diff_abs = round(abs(F_cap_CT_left - F_cap_DC_left), 4)
        # Find absolute difference between right CDFs of CT and DC datasets
        right_diff_abs = round(abs(F_cap_CT_right - F_cap_DC_right), 4)
        
        # Build each row to be appended to the KS Test Table 
        row = [x, F_cap_CT_left, F_cap_CT_right, F_cap_DC_left, F_cap_DC_right, left_diff_abs, right_diff_abs]
        row_dict = dict(zip(KS_Test_cols, row))
        row_list.append(row_dict)

    # Build KS Test Table (represented as a dataframe)
    KS_Test_df = pd.DataFrame(row_list, columns=KS_Test_cols)
    print("{1} 2c) 2-Sample KS Test using 'CT {0}' and 'DC {0}' columns {1}".format(column_type, 20*"-"))
    print(KS_Test_df)

    # Calculate KS statistic
    max_diff_x = []
    d_right = KS_Test_df.iloc[KS_Test_df['right_diff_abs'].idxmax(axis=1)][['x', 'right_diff_abs']]
    d_left = KS_Test_df.iloc[KS_Test_df['left_diff_abs'].idxmax(axis=1)][['x', 'left_diff_abs']]
    if d_right['right_diff_abs'] == d_left['left_diff_abs']:
        print("KS Statistic is {0} at x = {1} and {2}".format(d_right['right_diff_abs'], d_left['x'], d_right['x']))
        max_diff_x.append(d_right['x'])
        max_diff_x.append(d_left['x'])
    elif d_right['right_diff_abs'] > d_left['left_diff_abs']:
        print("KS Statistic is {0} at x = {1}".format(d_right['right_diff_abs'], d_right['x']))
        max_diff_x.append(d_right['x'])
    else:
        print("KS Statistic is {0} at x = {1}".format(d_left['left_diff_abs'], d_left['x']))
        max_diff_x.append(d_left['x'])
    
    # Reject/Accept Null Hypothesis based on calculated KS Statistic d and given threshold=0.05
    d = max(d_right['right_diff_abs'], d_left['left_diff_abs'])
    critical_value = 0.05
    hypothesis_type = 'confirmed positive cases' if column_type == 'confirmed' else column_type

    if d > critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the distribution of daily {0} is same in both CT and DC, as KS Statistic d = {1} exceeds threshold {2}".format(hypothesis_type, d, critical_value))
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the distribution of daily {0} is same in both CT and DC, as KS Statistic d = {1} does not exceed threshold {2}".format(hypothesis_type, d, critical_value))
    
    # Validate that calculated KS Statistic is same as the one obtained from the scipy KS Function
    # print("KS Statistic from scipy's KS Function = {0}".format(ks_2samp(CT_sorted_df[CT_col_name], DC_sorted_df[DC_col_name]).statistic))
    
    # Plot KS Test eCDF
    plot_KS_2_Sample_eCDF(CT_sorted_df, CT_col_name, DC_sorted_df, DC_col_name, max_diff_x, column_type.capitalize())
    print()
