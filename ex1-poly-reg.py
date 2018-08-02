import numpy as np
import random 
import matplotlib.pyplot as plt
import basicplot as bp


learning_rate = 0.0001415
iteration = 10000

def main():

    # Dataset
    x = np.arange(0, 20)
    y = (3 * x**2) + (11 * x) + 3

    # Plot data
    #bp.plot(x, y, "3x^2 + 11x + 3")
    
    # Training
    init_theta = np.asarray([0, 0, 0])

    cost_list, resulting_curve = gradientDescent(init_theta, x, y)
    print(cost_list)
    plt.plot(cost_list)
    plt.show()


    x = np.arange(0, 100)
    y = (3 * x**2) + (11 * x) + 3
    plt.plot(predictor(resulting_curve, x), label="pred")
    plt.plot(y, label="orig")
    plt.legend()
    plt.show()

    print(resulting_curve)

    

def computeCost(theta, x, y):
    m = len(y)
    cost = (1/(2*m)) * np.sum(np.square(predictor(theta, x) - y))
    return cost

def gradientDescent(init_theta, x, y):
    theta = init_theta
    cost_list = []
    m = len(y)
    #x_right = 
    for i in range(iteration):
        cost_list.append(computeCost(theta,x,y))
        y_pred = predictor(theta, x)
        del_sum = 0
        for j in range(m):
            x_cur = np.asarray([1, x[j], (x[j])**2])
            del_cur = (y[j] - y_pred[j]) * x_cur

            del_sum = del_sum + del_cur

        del_sum = del_sum / (2*m)
        #print(del_sum)
        theta = theta + learning_rate * del_sum

        #print(theta)

    print(theta)
    return cost_list, theta

def predictor(theta, x):
    return theta[0] + theta[1] * x + theta[2] * np.square(x)

main()