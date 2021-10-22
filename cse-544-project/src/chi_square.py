import scipy.stats

date_col_us_all = 'Date'
month_col_flights = 'MONTH'
year_col_flights = 'YEAR'

def chi_square_test(matrix, alpha=0.05):
    row_totals = []
    col_totals = [0] * len(matrix[0])
    total = 0

    for row in matrix:
        row_sum = 0
        for (index, col) in enumerate(row):
            row_sum += col
            col_totals[index] += col
        row_totals.append(row_sum)
        total += row_sum

    f_obs = []
    f_exp = []
    for (row_index, row) in enumerate(matrix):
        for (col_index, col) in enumerate(row):
            expected = col_totals[col_index] * (row_totals[row_index] / total)
            f_obs.append(col)
            f_exp.append(expected)
    # Note here that by default scipy takes dof to be k-1 where k is the number of cells.
    # We modify that behaviour to be consistent with that taught in class i.e (rows-1) * (cols-1)
    (chi_sq_statistic, p_value) = scipy.stats.chisquare(f_obs, f_exp,
                                                        (len(f_obs)-1) - (len(matrix) - 1) * (len(matrix[0]) - 1))
    if p_value < alpha:
        return p_value, False
    return p_value, True

# H_0: covid case count is independent of number of flight cancellations
# H_1: covid case count is dependent of number of flight cancellations
def perform_chi_square_test(min_month_NY, max_month_NY, flight_data):

    flights_data_NY = flight_data[
        (flight_data['ORIGIN_STATE_ABR'] == 'NY') | (flight_data['DEST_STATE_ABR'] == 'NY')]

    # X - Data for month with least average daily covid cases
    min_month, min_year = int(min_month_NY.split(' ')[0]), int(min_month_NY.split(' ')[1])
    X = flights_data_NY[
        (flights_data_NY[month_col_flights] == min_month) & (flights_data_NY[year_col_flights] == min_year)]

    # Y - Data for month with highest average daily covid cases
    max_month, max_year = int(max_month_NY.split(' ')[0]), int(max_month_NY.split(' ')[1])
    Y = flights_data_NY[
        (flights_data_NY[month_col_flights] == max_month) & (flights_data_NY[year_col_flights] == max_year)]

    min_month_cancellations = len(X[X['CANCELLED'] == 1])
    min_month_on_time = len(X[X['CANCELLED'] == 0])

    max_month_cancellations = len(Y[Y['CANCELLED'] == 1])
    max_month_on_time = len(Y[Y['CANCELLED'] == 0])

    p_value, hyp_decision = chi_square_test([[min_month_cancellations, max_month_cancellations], [min_month_on_time, max_month_on_time]])
    print("\n{0} Exploratory X Dataset Inference 2: Chi-Square Test {0}".format(20*"-"))
    print ("Performed chi square test with\n"
       "H_0: covid case count is independent of number of flight cancellations\n"
       "H_1: covid case count is dependent of number of flight cancellations\n"
       "p-value=%s, %s H_0\n" % (p_value, "Accept" if hyp_decision is True else "Reject"))