import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def compute_cost(x, y, w, b):
    total_cost = 0
    m = len(x)

    for i in range(m):
        f = w * x[i] + b

        cost = (f - y[i])**2

        total_cost += cost

        return 1/(2*m) * total_cost
    

def compute_gradient(x, y, w, b):
    djdw = 0
    djdb = 0

    m = len(x)

    for i in range(m):
        f = w * x[i] + b

        djdw += 1 / m * (f - y[i]) * x[i]

        djdb += 1 / m * (f - y[i]) 

    return djdw, djdb 
    
def gradient_descent(x, y, w_in, b_in, alpha, iterations, cost_function, gradient_function):
    djdw = 0
    djdb = 0
    J_history = []
    p_history = []
    m = len(x)
    w = w_in
    b = b_in

    for i in range(iterations):
        djdw, djdb = compute_gradient(x,y,w,b)

        w = w - alpha * djdw
        b = b - alpha * djdb 

        if i<100000:     
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        if i% math.ceil(iterations/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {djdw: 0.3e}, dj_db: {djdb: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")