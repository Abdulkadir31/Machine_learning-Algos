from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')



def best_fit_slope_intercept(X, Y):
    m = ( mean(X)* mean(Y) - mean(X*Y) )/( mean(X)**2 - mean(X*X))
    c = mean(Y) - m*mean(X)
    return m ,c

def squared_error(y_origin , y_line):
    return sum((y_line - y_origin)**2)

def coefficient_of_determination(y_origin,y_line):
    y_mean_line =[mean(y_origin) for y in y_origin]
    squared_error_reg = squared_error(y_origin,y_line)
    squared_error_Ymean = squared_error(y_origin,y_mean_line)
    return 1 - (squared_error_reg/squared_error_Ymean)

def create_dataset(how_much, variance, step=2, correlation='False'):
    val = 1
    y = []
    for i in range(how_much):
        y_val = val + random.randrange(-variance,variance)
        y.append(y_val)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    x = [i for i in range(len(y))]

    return(np.array(x,dtype = np.float64),np.array(y,dtype=np.float64))


x,y = create_dataset(40,40,2,'pos')


m ,c= best_fit_slope_intercept(x,y)

regression_line = [ (m*x)+ c for x in x]



r_squared = coefficient_of_determination(y,regression_line)
print(r_squared)
plt.scatter(x,y)
plt.plot(x,regression_line)
plt.show()