import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	np_X = np.array(X)
	np_Y = np.array(y)
	np_X_transpose = np.transpose(np_X)
	theta = (np.linalg.inv(np_X_transpose @ np_X)) @ (np_X_transpose @ np_Y)
	return theta