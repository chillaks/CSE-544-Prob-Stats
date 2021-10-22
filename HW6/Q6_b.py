import math
import pandas as pd

# 10 different instances of observations of set w containing n=500 samples each. Column Oi(i = 0...9) is the ith instance of observations
columns = ['O'+ str(i) for i in range(10)]
w_dataset = pd.read_csv('./datasets/q6.csv', names=columns)
# print(w_dataset)

# prior probability list
p = [0.1, 0.3, 0.5, 0.8]
# given mu, var
mu, var = 0.5, 1

def MAP_descision(w, prior_prob):
    # return C=0 when (var/2*mu) * ln(p/(1-p)) >= sigma(w_i), and C=1 otherwise
    if (var / (2 * mu) * math.log(prior_prob / (1 - prior_prob))) >= w.sum():
        C = 0
    else:
        C = 1
    return C

for prior_prob in p:
    res_H_list = []
    for column in columns:
        # Determine selected hypothesis C(0,1) for each instance of observation for w, for each of the given prior probabilities
        selected_H = MAP_descision(w_dataset[column].values, prior_prob)
        res_H_list.append(str(selected_H))
    print('For P(H0) = {0}, the hypotheses selected are :: {1}'.format(prior_prob, " ".join(res_H_list)))

