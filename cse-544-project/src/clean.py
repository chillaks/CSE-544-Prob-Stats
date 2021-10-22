import pandas
import math
import numpy as np

def get_cleaned_data(filename, us_all=False, drop_outliers=False):
    if not us_all:
        data_raw = pandas.read_csv(filename)
    else:
        data_raw = pandas.read_csv(filename).set_index('State').transpose().reset_index().rename(columns={'index': 'Date'})
        data_raw.columns.name = None
    outliers = set()
    daily_data = pandas.DataFrame()

    for column in data_raw:
        if column == 'Date':
            daily_data[column] = data_raw[column]
            continue
        
        daily_column = data_raw[column].diff().fillna(data_raw.iloc[0][column])
        daily_data[column] = daily_column
        sorted_column_data = np.array(daily_column.sort_values())

        Q3_index = math.ceil(75/100 * len(sorted_column_data))
        Q1_index = math.ceil(25/100 * len(sorted_column_data))
        alpha = 1.5
        IQR = sorted_column_data[Q3_index] - sorted_column_data[Q1_index]
        upper_threshold = sorted_column_data[Q3_index] + alpha * IQR
        lower_threshold = sorted_column_data[Q1_index] - alpha * IQR

        column_outliers = daily_column[(daily_column != 0) & ((daily_column < lower_threshold) | (daily_column > upper_threshold))].index
        outliers = outliers.union(column_outliers)
        print ("\nColumn: %s \nDetected %s outlier rows \nIQR = %s \nlower threshold = %s \nupper threshold = %s" % (
        column, len(column_outliers), IQR, lower_threshold, upper_threshold))
    
    if drop_outliers:
        data_raw.drop(outliers, inplace=True)
        daily_data.drop(outliers, inplace=True)

    return data_raw, daily_data
