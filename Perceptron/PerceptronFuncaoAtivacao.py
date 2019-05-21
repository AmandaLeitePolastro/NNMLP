import pandas as pd
import numpy as np
import random as rd
from scipy.signal import lfilter
import matplotlib
from matplotlib import pyplot as plt

#signals generation
n = 10000
n_samples = np.linspace(0,n-1,n) 

#white noise generation
s_data = np.random.uniform(-0.9, 0.9, n) #uniform white noise with 10000 samples between -0.9 and 0.9

#Hammerstein system that adds nonlinearity and memory in the white noise (s_data)
u_data = np.arctan(s_data) #addition of nonlinearity in white noise
x_data = lfilter([1, 0.6, 0, 0, 0, 0, 0.2], 1, u_data) #addition of memory in the white noise


#classical perceptron regarding activation function derivative
def activation_function_derivative_perceptron(input_data, desired_data, bias, lr):
    
    #activation function parameters
    a = 1
    b = 1
    c = b/a
    
    #perceptron parameters
    wind = 3
    winx = 2
    
    xm = np.zeros(winx)
    x_data = np.append(xm, input_data) 
    
    w = np.zeros(wind)
    w_temp = np.append(bias, w) #dim = 4
   
    y = []
   
    
    for k in range (0, len(input_data)):
        
        w = w_temp #dim = 4
        
        #x_data regards a time window with 3 samples
        x_temp = x_data[k:wind+k] #dim = 3
        x_temp2 = x_temp[::-1] #reverse x_temp
        x_temp3 = np.append(1, x_temp2) #dim = 4 (3 samples + 1 for bias)
        
        #v is the dot product between w_temp and x_temp
        #y is the estimated value by the perceptron
        v = np.dot(w,x_temp3)
        y_temp = np.tanh(v) #activation function
        y.append(y_temp)

        #distance between the real and estimated values
        error = desired_data[k] - y_temp

        #atualization of the weights
        if y_temp == desired_data[k]:
            w_out = w
        else:
            #the weight regards the activation function derivative 
            w_out = w + lr*error*c*(a - y_temp)*(a + y_temp)*x_temp3
    
        w_temp = w_out
        
    return w_out, y



#classical perceptron without activation function derivative
def perceptron(input_data, desired_data, bias, lr):
    
    wind = 3
    winx = 2
    xm = np.zeros(winx)
    
    x_data = np.append(xm, input_data) 
    
    w = np.zeros(wind)
    w_temp = np.append(bias, w) #dim = 4
   
    y = []
   
    
    for k in range (0, len(input_data)):
        
        w = w_temp #dim = 4
        
        x_temp = x_data[k:wind+k] #dim = 3
        x_temp2 = x_temp[::-1] #reverse x_temp
        x_temp3 = np.append(1, x_temp2) #dim = 4
        
        #y_temp is the dot product between w_temp and x_temp
        y_temp = np.tanh(np.dot(w,x_temp3))
        y.append(y_temp)

        if y_temp == s_data[k]:
            w_out = w
        else:
            w_out = w + lr*(desired_data[k] - y_temp)*x_temp3
    
        w_temp = w_out
        
    return w_out, y


#EQM function
def EQM(desired_data, y_data):
    
    esignal = []
    
    for k in range(0,len(y_data)):
        esample = (y_data[k] - desired_data[k])**2
        esignal.append(esample)
    
    eqm_temp = np.sum(esignal)
    eqm = eqm_temp/len(y_data)

    return eqm


#training
wtheta_perceptron, y_perceptron = perceptron(x_data, s_data, 0.5, 0.002)
wtheta_perceptron_af, y_perceptron_af = activation_function_derivative_perceptron(x_data, s_data, 0.5, 0.002)

#eqm
eqm_perceptron = EQM(s_data, y_perceptron)
eqm_perceptron_af = EQM(s_data, y_perceptron_af)


#plot of the results
fig1 = plt.figure(figsize=(8.0, 6.0))
ax = fig1.gca()

plt.plot(n_samples[9900:10000], s_data[9900:10000], color='black', label="s(n)")
plt.plot(n_samples[9900:10000], y_perceptron[9900:10000], color='blue', label="y(n)")
plt.plot(n_samples[9900:10000], y_perceptron_af[9900:10000], color='red', label="y_af(n)")

plt.xlabel('n')
plt.ylabel('s(n), y(n), y_af(n)')
plt.legend(loc='upper right')

plt.grid()
plt.show()

