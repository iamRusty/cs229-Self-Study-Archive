import os
import numpy as np
import matplotlib.pyplot as plt

num_iters = 1500
learning_rate = 0.01

def main():

	# Load data
	filename = "ex1data1.txt"
	x, y = np.loadtxt(filename, delimiter=',', unpack=True)

	# Plot data
	plt.scatter(x,y, marker='x', color='red', linewidth=0.7)
	plt.xlabel("Population of City in 10,000s")
	plt.ylabel("Profit in $10,000s")
	plt.show()
	plt.gcf().clear()

	# Force x and y to be vectors
	m = len(y)
	x.shape = (m, 1)
	y.shape = (m, 1)

	# Gradient Descent Initialization
	temp = np.ones(m)					# bias
	temp.shape = (m, 1)
	x = np.append(temp, x, axis=1) 		# axis = 1 append in parallel
	x = x.T								# to be consistent with w.Tx + b
	theta = np.zeros((2, 1))			# initialize weights
	
	# Gradient Descent Proper
	J_hist, theta_new = gradientDescent(theta, x, y, learning_rate, num_iters)

	# Plot cost history
	plt.plot(J_hist)
	plt.xlabel("Iteration number")
	plt.ylabel("Cost/error (normalized)")
	plt.show()
	plt.gcf().clear()

	# Plot data and prediction line
	x1, y1 = np.loadtxt(filename, delimiter=',', unpack=True)
	linex = list(range(int(x1.min()), int(x1.max()+1)))
	liney = []
	for i in range (0, len(linex)):
		liney.append(theta_new[0] + theta_new[1]*linex[i])

	plt.scatter(x1,y1, marker='x', color='red', linewidth=0.7)
	plt.plot(linex, liney)
	plt.xlabel("Population of City in 10,000s")
	plt.ylabel("Profit in $10,000s")
	plt.show()


def computeCost(theta, x, y):
	m 		= len(y)
	h 		= np.dot(theta.T, x)
	diff 	= h - y.T
	sq_err 	= np.dot(diff, diff.T)
	J 		= sq_err/(2*m)
	J 		= np.asscalar(J)
	return J

def gradientDescent(theta, x, y, learning_rate, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters,1))
	for i in range(num_iters):

		diff = np.dot(theta.T, x) - y.T
		delta = (1/m)*np.dot(diff, x.T)
		theta = theta - learning_rate * delta.T
		hello = computeCost(theta, x, y)
	
		print("Iteration %d - Cost: %f" % (i + 1, hello))

		# Save cost/error history
		J_history[i] = hello

	return J_history, theta

main()