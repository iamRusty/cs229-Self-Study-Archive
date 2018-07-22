import numpy as np
import matplotlib.pyplot as plt
import os


filename = "ex2data1.txt"
learning_rate = 0.0015
num_iters = 100000

def main():

    # Load dataset
    exam1, exam2, result = np.loadtxt(filename, delimiter=',', unpack=True)

    """
    # Plot data
    color_vector = ["red" if i == 0 else "blue" for i in result]
    plt.scatter(exam1, exam2, c=color_vector, marker='x', linewidth=0.7)
    plt.xlabel("Exam 1 scores")
    plt.ylabel("Exam 2 scores")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()   
    """

    # Prepare dataset
    m = exam1.shape[0]
    intercept = np.ones((1,m))
    x = np.concatenate((intercept, exam1.reshape(1,m), exam2.reshape(1,m)), axis=0)
    y = result.reshape(m, 1)
    initial_theta = np.zeros((x.shape[0], 1))
    

    cost_history, theta = gradientDescent(x, y, initial_theta)
    plt.plot(cost_history)
    plt.show()

    while(1):
        exam1_s = input("Exam 1: ")
        exam2_s = input("Exam 2: ")

        predict(theta, exam1_s, exam2_s)

    #cost, delta = costFunction(x, y, initial_theta)
    #print(cost, initial_theta - learning_rate * delta )

def prediction(theta, x, y):
    #print(theta.T)
    hello2 = sigmoid(np.dot(theta.T, x)).T
    hello = []
    hello = [0 if i < 0.5 else 1 for i in hello2]
    hello = np.abs(hello - y.T)
    print(np.sum(hello))

def predict(theta, exam1, exam2):
    x = np.asarray([1, int(exam1), int(exam2)])
    z = np.dot(theta.T, x)
    temp = sigmoid(z)
    if (temp < 0.5):
        print("0")
    else:
        print("1")

def gradientDescent(x, y, theta):
    cost_history = []
    print("Hello World")
    for i in range(num_iters):
        cost, delta = costFunction(x, y, theta)
        theta = theta - learning_rate * delta
        cost_history.append(cost)        
    
        #print("Iter number: " + str(i) + " Cost: " + str(cost))
        #prediction(theta, x, y)

    prediction(theta, x, y)
    return cost_history, theta

def costFunction(x,y, theta):
    # delta or grad should have the same dimension as theta
    m = y.shape[0]
    z = np.dot(theta.T, x)
    h = sigmoid(z)
    J = (-1) * ( np.dot(np.log(h), y) + np.dot(np.log(1-h), (1-y)) ) 
    J_norm = np.asscalar(J)/m

    delta = np.dot(x, (h - y.T).T)
    delta_norm = delta/m

    return J_norm, delta_norm

def sigmoid(z):
    return 1/(1+np.exp(-z))


main()