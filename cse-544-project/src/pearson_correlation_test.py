import math
from datetime import date, timedelta

import pandas as pd

us_confirmed_data_file_path = '../data/US-all/US_confirmed.csv'
us_flight_data_file_path = '../data/X_flights_cancellation/jantojun2020.csv'


class PearsonsTest:

    def __init__(self, X, Y):
        self.__X = X
        self.__Y = Y
        self.__sample_mean_X = sum(self.__X) / len(self.__X)
        self.__sample_mean_Y = sum(self.__Y) / len(self.__Y)

    def __get_sample_variance(self, data, sample_mean):
        sample_variance = 0
        for d in data:
            sample_variance += pow(d - sample_mean, 2)
        return sample_variance

    def __get_co_variance(self):
        co_variance = 0
        for i in range(0, len(self.__X)):
            co_variance += (self.__X[i] - self.__sample_mean_X) * (self.__Y[i] - self.__sample_mean_Y)
        return co_variance

    def __get_pearsons_coefficent(self):
        sample_variance_x = self.__get_sample_variance(self.__X, self.__sample_mean_X)
        sample_variance_y = self.__get_sample_variance(self.__Y, self.__sample_mean_Y)
        return self.__get_co_variance() / math.sqrt(sample_variance_x * sample_variance_y)

    def perform_test(self):
        pearson_coefficent = self.__get_pearsons_coefficent()
        if pearson_coefficent > 0.5:
            return "Positive correlation", pearson_coefficent
        elif pearson_coefficent < -0.5:
            return "Negative correlation", pearson_coefficent
        else:
            return "No correlation", pearson_coefficent



# fetches per day cancelled flights with source or destination as state
def get_cancelled_flights_df(state, flights_df):
    flights_df = flights_df[['YEAR', 'MONTH', 'DAY_OF_MONTH', 'CANCELLED']]
    return flights_df.groupby(['YEAR', 'MONTH', 'DAY_OF_MONTH'], as_index=False).agg({'CANCELLED': ['sum']})


# generates datasets X, Y needed for pearsons test
# X = number of new cases for a given state
# Y = number of flights cancelled for a given sate
# if X[index] has number of new cases on day d then Y[index] has number of flights cancelled on the same day d
def get_datasets_for_pearsons_test(daily_cases_df, flight_df, date_from, date_to):
    x = []
    y = []
    for dt in pd.date_range(date_from, date_to + timedelta(days=1)):
        date_column = dt.strftime("%Y-%m-%d")
        if (not flight_df.loc[(flight_df['DAY_OF_MONTH'] == dt.day) & (flight_df['MONTH'] == dt.month) & (
                flight_df['YEAR'] == dt.year)].empty) and (not daily_cases_df[daily_cases_df['Date'] == date_column].empty):
            x.append(daily_cases_df[daily_cases_df['Date'] == date_column].iloc[0, 1])
            y.append(flight_df[(flight_df['DAY_OF_MONTH'] == dt.day) & (flight_df['MONTH'] == dt.month) & (
                        flight_df['YEAR'] == dt.year)].iloc[0][3])
    return x, y


# dates should be between 22 Jan 2020 and 30 Jun 2020
def perform_pearsons_correlation_test(state, date_from, date_to, flights_df, cases_df):
    print("\n{0} Exploratory X Dataset Inference 1: Pearson's Coefficient to test for Correlation {0}".format(20*"-"))
    cancelled_flights = get_cancelled_flights_df(state, flights_df)
    x, y = get_datasets_for_pearsons_test(cases_df, cancelled_flights, date_from, date_to)
    test = PearsonsTest(x, y)
    type_of_correlation, coefficient = test.perform_test()
    print('For State {} between dates {} and {}:\nLet X = Daily new cases and Y = Total flights cancelled\nUsing '
          'Pearsons Correlation Test we get:\nType of '
          'correlation : {}, Correlation coefficient : {}'.format(state, date_from, date_to, type_of_correlation,
                                                                  coefficient))
