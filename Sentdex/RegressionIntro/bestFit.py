from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style

style.use("fivethirtyeight")

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
	m = ( ( ( mean(xs) * mean(ys) ) - mean(xs*ys) ) /
		( (mean(xs) ** 2) - mean(xs**2) ) 
		)
	b = mean(ys) - m * mean(xs)

	return m, b

# Distance between the line in question and the points
def squared_error(ys_orig, ys_line):

	return sum( (ys_line - ys_orig)**2 )


def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [ mean(ys_orig) for y in ys_orig ]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)

	return 1 - (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x) + b for x in xs]

# Predict x = 8
# predict_x = 8
# predict_y = (m*predict_x) + b

# plt.scatter(xs, ys)
# plt.plot(xs, regression_line)
# plt.show()

r_squared = coefficient_of_determination(ys, regression_line)

# Anything above 0 means the regression is more accurate.
# squared error and cofficient of determination is a way to calculate 
# how good of a fit the best fit line is
print r_squared





