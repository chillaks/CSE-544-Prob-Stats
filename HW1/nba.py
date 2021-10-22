import numpy as np

# seed with any +ve integer to get same results over all executions of this script.
# comment below line in case above behavior is not desired
np.random.seed(1)
# number of trials
n = [1000, 10000, 100000, 1000000, 10000000]

# list to store the number of times we simulated LAC winning first 3 games out of 4 over each of the 
# n number of trials
lac3_num_list = []

# part a)
for size in n:
	# For part a), we need to find P(LAC 3 - DEN 1), therefore number of independant experiments
	# for each trial is 3 + 1 = 4
	trials = 4
	# given, probability of winning for LAC is 0.5
	p = 0.5

	# number of outcomes where LAC have won 3 times
	num_lac3 = sum([1 for num_lac_wins in np.random.binomial(trials, p, size) if num_lac_wins == 3])
	lac3_num_list.append(num_lac3)

	# Avg. Probability of LAC winning 3 games out of 4, sampled over 'size' number of trials
	p_lac3 = num_lac3 / size

	print("For N = {0}, the simulated value for part (a) is {1}".format(size, p_lac3))
print()

# part c)
i = 0
for num_lac3 in lac3_num_list:
	# Given LAC 3 - DEN 1, we need to find P(DEN4 - LAC3). This is possible only if DEN win all of the 
	# last 3 games, that is, LAC lose all. Hence, out of 3 independant experiments (games), we need to
	# count the number times we got 0 successes for LAC
	trials, p = 3, 0.5

	# We derived that LAC are 3-1 up num_lac3 times, which becomes the size we use. Hence, number of times DEN
	# won, i.e, LAC lost all of the next 3 games becomes
	num_den4_lac3 = sum([1 for num_lac_wins in np.random.binomial(trials, p, num_lac3) if num_lac_wins == 0])

	p_den4_given_lac3 = num_den4_lac3 / num_lac3

	print("For N = {0}, the simulated value for part (c) is {1}".format(n[i], p_den4_given_lac3))
	i = i + 1
print()

# part e)
for size in n:
	# First 2 games played at LAC home, and LAC has 0.75 chances of winning. The binomial distribution of all
	# possible outcomes of LAC winning over each of n number of trials is derived by
	bin_dist_lac_wins_home = np.random.binomial(2, 0.75, size)
	# Next 2 games are played away at DEN, and LAC has 0.25 chances of winning. Similarly,
	bin_dist_lac_wins_away = np.random.binomial(2, 0.25, size)

	# number of outcomes where LAC won 3 games out of 4, given first 2 games were home and next 2 away
	num_lac3 = sum([1 for i in range(size) if bin_dist_lac_wins_home[i] + bin_dist_lac_wins_away[i] == 3])

	# Now, we need to derive the number of outcomes where LAC lost the next 3 games, given they were leading 3-1.
	# Hence, the sampling size becomes the number of outcomes where LAC are leading 3-1 in the first 4 games
	bin_dist_lac_win_g5 = np.random.binomial(1, 0.75, num_lac3)  # LAC Home
	bin_dist_lac_win_g6 = np.random.binomial(1, 0.25, num_lac3)  # DEN Home
	bin_dist_lac_win_g7 = np.random.binomial(1, 0.75, num_lac3)  # LAC Home

	# Possible number of outcomes where DEN won the next 3 games is the sum of the distributions where LAC lost each of those games
	num_den4_lac3 = sum([1 for i in range(num_lac3) if bin_dist_lac_win_g5[i] + bin_dist_lac_win_g6[i] + bin_dist_lac_win_g7[i] == 0])

	p_den4_given_lac3 = num_den4_lac3 / num_lac3

	print("For N = {0}, the simulated value for part (e) is {1}".format(size, p_den4_given_lac3))

