# -*- coding: utf-8 -*-
"""

Code to calculate KS 1 sample test and accept/reject hypothesis(2c)
Code to calculate Permutation test and accept/reject hypothesis(2c)
Takes cleaned state 4 dataframe as input 
Input to be passed to KS_1_sample_main and Permutation_main
"""
import copy
import pandas as pd
import numpy as np
from decimal import *
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import binom
import matplotlib.pyplot as plt
import math
import clean


# Calculate CDF for the number of cases/deaths on each day
def get_cdf_list(num_samples):
    cdf_list = []
    cumulative_pr = 0
    for _ in range(num_samples):
        cumulative_pr += 1 / num_samples
        cdf_list.append(cumulative_pr)
    return cdf_list

#Calculate max cdf to the left of the point
def get_left_cdf(state_df, col_name, x, eCDF_col):
    if state_df[col_name].max() < x:
        return 1
    elif state_df[col_name].min() > x:
        return 0
    else:
        left_cdf = state_df.loc[state_df[col_name] < x, eCDF_col]
        F_cap_left = 0.0 if left_cdf.empty else left_cdf.max()
        return F_cap_left

#Calculate min cdf to the right of the point
def get_right_cdf(state_df, col_name, x, eCDF_col):
    if state_df[col_name].max() < x:
        return 1
    elif state_df[col_name].min() > x:
        return 0
    else:
        right_cdf = state_df.loc[state_df[col_name] >= x, eCDF_col]
        F_cap_right = 0.0 if right_cdf.empty else right_cdf.min()
        return F_cap_right
 
#Calculate sample mean for given data and column
def sample_mean(data, CT_col_name):
    x_points = data[CT_col_name].to_list()
    total = 0
    mean = 0
    for x in x_points:
        total = total + x
    mean = total/ len(data)
    
    return mean

#Calculate sample variance for given data and column and mean
def sample_variance(data, sample_mean, CT_col_name):
    x_points = data[CT_col_name].to_list()
    total = 0
    for x in x_points:
        total = total + pow((x-sample_mean),2)
    variance = total/len(data)
    
    return variance

#Calculates Poisson MME Parameters
def poisson_para(mean):
    lambda_mme = mean
    return lambda_mme

#Calculates Geometric MME Parameters
def geometric_para(mean):
    p_mme = 1/mean
    return p_mme

#Calculates Binomial MME Parameters
def binomial_para(mean,variance):
    n_mme = pow(mean,2)/(mean - variance)
    p_mme = mean/n_mme
    
    return n_mme,p_mme

def plot_KS_1_Sample_eCDF(DC_df, DC_col_name, max_diff_x, x_label, distribution_type):
    
    y_ecdf_mme = DC_df['DC_eCDF_mme'].to_numpy()
    x_points_DC = DC_df[DC_col_name].to_numpy()
    y_ecdf_DC = DC_df['DC_eCDF'].to_numpy()

    plt.figure('KS 1-Sample Test eCDF', figsize=(6,6))
    plt.step(x_points_DC, y_ecdf_mme, where='post', lw = 1.5, label=distribution_type+ ' CDF')
    plt.step(x_points_DC, y_ecdf_DC, where='post', lw = 1.5, label='DC eCDF')
    for x in max_diff_x:
        plt.axvline(x, linestyle="dashed", lw=1)

    plt.xlabel(x_label)
    plt.ylabel('eCDF')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



#Calculates KS Statistic Values
def calc_KS_1_sample_test(x_points, parameters_list, data, distribution_name, column_type , column_name):
    dict_name = 'KS_' + distribution_name +'_cols'
    dict_name = ['x', 'F_cap_x', 'F_cap_DC_left', 'F_cap_DC_right', 'left_diff_abs', 'right_diff_abs']
    row_list = []
    for x in x_points:
        if(distribution_name == 'binomial'):
            #Find cdf of binomial at given point x
            F_cap_x = binom.cdf(x, parameters_list[0], parameters_list[1])
        if(distribution_name == 'poisson'):
            #Find cdf of poisson at given point x
            F_cap_x = poisson.cdf(x, parameters_list[0])
        if(distribution_name == 'geometric'):
            #Find cdf of geometric at given point x
            F_cap_x = geom.cdf(x, parameters_list[0])
        # Find CDF to the left of point x in the sorted DC dataset
        F_cap_DC_left = get_left_cdf(data ,column_name, x, 'DC_eCDF')
        # Find CDF to the right of point x in the sorted DC dataset
        F_cap_DC_right = get_right_cdf(data, column_name, x, 'DC_eCDF')
        # Find absolute difference between left CDFs of x points and DC datasets
        left_diff_abs = round(abs(F_cap_x - F_cap_DC_left), 4)
        # Find absolute difference between right CDFs of x points and DC datasets
        right_diff_abs = round(abs(F_cap_x - F_cap_DC_right), 4)
    
        row = [x, F_cap_x, F_cap_DC_left, F_cap_DC_right, left_diff_abs, right_diff_abs]
        row_dict = dict(zip(dict_name, row))
        row_list.append(row_dict)
    
    # Build KS Test Table (represented as a dataframe)    
    df_name = 'KS_' + distribution_name +'_df'
    df_name = pd.DataFrame(row_list, columns=dict_name)
    
    # Calculate KS statistic value
    max_diff_x = []
    d_right = df_name.iloc[df_name['right_diff_abs'].idxmax(axis=1)][['x', 'right_diff_abs']]
    d_left = df_name.iloc[df_name['left_diff_abs'].idxmax(axis=1)][['x', 'left_diff_abs']]
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
        print("Rejected Null Hypothesis: We reject the hypothesis that the distribution of daily {0} in DC is {3}, as KS Statistic d = {1} exceeds threshold {2}".format(hypothesis_type, d, critical_value, distribution_name))
        print()
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the distribution of daily {0} is same in both CT and DC, as KS Statistic d = {1} does not exceed threshold {2}".format(hypothesis_type, d, critical_value))
        print()
        
        
    return max_diff_x

#Generates required data before calling calculation of KS 1 sample test statistic 
def KS_1_sample_test(states_data,column_type, distribution_type):
    CT_col_name = 'CT ' + column_type
    DC_col_name = 'DC ' + column_type

    # Split the dataset per state and sort the 2 state-specific columns on which we need to perform the KS Test (#cases/#confirmed)
    CT_sorted_df = states_data.loc[(states_data['Date'] >= '2020-10-01') & (states_data['Date'] <= '2020-12-31')][[CT_col_name]].sort_values(CT_col_name).reset_index(drop=True)
    DC_sorted_df = states_data.loc[(states_data['Date'] >= '2020-10-01') & (states_data['Date'] <= '2020-12-31')][[DC_col_name]].sort_values(DC_col_name).reset_index(drop=True)

    # Add a new column denoting the CDF at each point in the DC cases/deaths columns
    DC_sorted_df['DC_eCDF'] = get_cdf_list(DC_sorted_df.shape[0])
    
    # Find distinct datapoints and their corresponding CDFs at each of the points
    DC_distinct_df = DC_sorted_df.drop_duplicates(subset=DC_col_name, keep="last").reset_index(drop=True)
    
    # points for x column for KS test
    x_points = DC_distinct_df[DC_col_name].to_list()
    
    if (distribution_type == 'poisson'):
        #Poisson distribution
        mean = sample_mean(CT_sorted_df, CT_col_name)
        #MME parameters
        lambda_mme = poisson_para(mean)
        #Calls calculation of KS 1 sample statistic 
        max_diff_x = calc_KS_1_sample_test(x_points, [lambda_mme] , DC_distinct_df , 'poisson', column_type,DC_col_name ) 
        
        DC_sorted_df['DC_eCDF_mme'] = DC_sorted_df.apply(lambda row : poisson.cdf(row[DC_col_name], lambda_mme),axis =1)
        plot_KS_1_Sample_eCDF( DC_sorted_df, DC_col_name, max_diff_x, column_type.capitalize(),distribution_type)
        
        
    if (distribution_type == 'geometric'):
        #Geometric distribution
        mean = sample_mean(CT_sorted_df,CT_col_name)
        #MME parameters
        p_mme = geometric_para(mean)
        #Calls calculation of KS 1 sample statistic 
        max_diff_x = calc_KS_1_sample_test(x_points, [p_mme] , DC_distinct_df , 'geometric', column_type, DC_col_name)
        
        DC_sorted_df['DC_eCDF_mme'] = DC_sorted_df.apply(lambda row :  geom.cdf(row[DC_col_name], p_mme),axis =1)
        plot_KS_1_Sample_eCDF( DC_sorted_df, DC_col_name, max_diff_x, column_type.capitalize(),distribution_type)
    
    if (distribution_type == 'binomial'):
        #Binomial distribution
        mean = sample_mean(CT_sorted_df,CT_col_name)
        variance = sample_variance(CT_sorted_df,mean,CT_col_name)
        #MME parameters
        n_mme, p_mme = binomial_para(mean,variance)
        #Calls calculation of KS 1 sample statistic c
        max_diff_x = calc_KS_1_sample_test(x_points, [n_mme, p_mme] , DC_distinct_df , 'binomial', column_type, DC_col_name)
        
        DC_sorted_df['DC_eCDF_mme'] = DC_sorted_df.apply(lambda row : binom.cdf(row[DC_col_name], n_mme,  p_mme),axis =1)
        plot_KS_1_Sample_eCDF(DC_sorted_df, DC_col_name, max_diff_x, column_type.capitalize(),distribution_type)

    
#Calculates p value for permutaiton test
def calc_permutation_test(sample_size, data_points, t_obs , len_points):
    outlier_count = 0
    for i in range(sample_size):
        #Generates a random array of data_points
        perm_data = np.random.permutation(data_points)
        CT_mean = 0
        DC_mean = 0
        for index in range(len(perm_data)):
            if index < len_points:
                #Calculates sum of X points
                CT_mean += float(perm_data[index])
            else:
                #Calculates sum of Y points
                DC_mean += float(perm_data[index])
        #Calculates mean of X and Y points
        CT_mean /= len_points
        DC_mean /= (len(data_points) - len_points)
        #Add 1 if difference in mean greater than T observed
        if abs(CT_mean - DC_mean) > t_obs:
            outlier_count += 1
    #Calculates and returns p value
    return outlier_count / sample_size

#Generates required data before calling calculation of permutation test p value
def Permutation_test(states_data,column_type):
    
    CT_col_name = 'CT ' + column_type
    DC_col_name = 'DC ' + column_type

    # Split the dataset per state and sort the 2 state-specific columns on which we need to perform the KS Test (#cases/#confirmed)
    CT_df = states_data.loc[(states_data['Date'] >= '2020-10-01') & (states_data['Date'] <= '2020-12-31')][[CT_col_name]]
    DC_df = states_data.loc[(states_data['Date'] >= '2020-10-01') & (states_data['Date'] <= '2020-12-31')][[DC_col_name]]
    
    #Calculates sample mean of X and Y points
    CT_mean = sample_mean(CT_df, CT_col_name)
    DC_mean = sample_mean(DC_df, DC_col_name)
    
    #Calculates T observed 
    T_obs = abs(CT_mean - DC_mean)
    
    #Converts dataframe points into list
    x_points = CT_df[CT_col_name].to_list()
    y_points = DC_df[DC_col_name].to_list()

    #Adds both lists to create a single list for random permutation
    data_points = x_points + y_points
    
    #Calls calculation of p value function with 1000 permutations
    p_value_1000 = calc_permutation_test(1000, data_points, T_obs , len(x_points))
    
    # Reject/Accept Null Hypothesis based on calculated p value and given threshold=0.05
    critical_value = 0.05
    hypothesis_type = 'confirmed positive cases' if column_type == 'confirmed' else column_type
    
    if p_value_1000 <= critical_value:
        print("Rejected Null Hypothesis: We reject the hypothesis that the distribution of daily {0} is same in both CT and DC, as Permutation test p-value = {1} does not exceed threshold {2}".format(hypothesis_type, p_value_1000, critical_value))
        print()
    else:
        print("Failed to reject Null Hypothesis: We accept the hypothesis that the distribution of daily {0} is same in both CT and DC, as Permutation test p-value = {1} exceeds threshold {2}".format(hypothesis_type, p_value_1000, critical_value))
        print()
        
#KS 1 sample test main function       
def KS_1_sample_main(states_data):
    print("{0} 2c) KS_1_sample test starts here {0}".format(20*"-"))
    KS_1_sample_test(states_data,'confirmed', 'poisson')
    KS_1_sample_test(states_data,'confirmed', 'geometric')
    KS_1_sample_test(states_data,'confirmed', 'binomial')
    KS_1_sample_test(states_data,'deaths', 'poisson')
    KS_1_sample_test(states_data,'deaths', 'geometric')
    KS_1_sample_test(states_data,'deaths', 'binomial')

#Permutation test main function       
def Permutation_main(states_data):
    print("{0} 2c) Permutation test starts here {0}".format(20*"-"))
    Permutation_test(states_data,'confirmed')
    Permutation_test(states_data,'deaths')
