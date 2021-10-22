import numpy as np
import matplotlib.pyplot as plt
import math


def predict_ewma(alpha, data):
    ewma = 0

    for i in range(len(data)):
        weight = pow(1 - alpha, i)
        ewma += weight * data[len(data) - 1 - i]
    return alpha * ewma

alpha_values = [0.5, 0.8]


def run_ewma_analysis(data):
    for alpha in alpha_values:
        fig, axs = plt.subplots(2, 2)
        count = 0

        for column in data:
            if column == 'Date':
                continue
            print()

            y_actual = []
            y_predicted = []
            x_axis = []
            MSE = 0
            MAPE = 0
            mape_count = 0
            for end in range(22, 29):
                end_date = '2020-08-%s' % end

                training_data = data[(data['Date'] >= '2020-08-01') & (data['Date'] < end_date)]

                prediction = predict_ewma(alpha, np.float_(np.array(training_data[column])))
                print("Predicting %s on %s for alpha=%s using EWMA(%s): %s" % (column, end_date, alpha, alpha, prediction))

                actual_value = data[data['Date'] == end_date][column].values[0]
                MSE += pow(prediction - actual_value, 2)
                if actual_value != 0:
                    MAPE += abs(prediction - actual_value) / actual_value * 100
                    mape_count += 1

                y_actual.append(actual_value)
                y_predicted.append(prediction)
                x_axis.append(end)

            MSE /= len(range(22, 29))
            MAPE /= mape_count
            print ("MSE for alpha = %s: %s" % (alpha, MSE))
            print ("MAPE for alpha = %s: %s" % (alpha, MAPE))
            plot = axs[math.floor(count / 2), count % 2]
            plot.set_title("EWMA(%s) for %s" % (alpha, column))
            plot.plot(x_axis, y_actual, label="actual data")
            plot.plot(x_axis, y_predicted, label="predicted data")
            plot.legend()
            count += 1
        plt.show()
