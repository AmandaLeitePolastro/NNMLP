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


#modified perceptron regarding activation function derivative
def activation_function_derivative_modified_perceptron(desired_data, input_data, lr):

    #activation function parameters
    ath = 1
    bth = 1
    cth = bth/ath

    #signal parameters and generation
    wind = 3
    winx = 2
    
    xm = np.zeros(winx) 
    x_data = np.append(xm, input_data)
 
    winy = 3
    y_input = np.zeros(winy)
    
    #weights initialization
    a = np.zeros(wind)
    b = np.zeros(wind)
    wtheta = np.append(a,b)
   
    for k in range(0,len(input_data)):
   
        y_temp = y_input[k:wind+k]
        y = y_temp[::-1] #reverse y_temp
        
        x_temp = x_data[k:wind+k]
        x = x_temp[::-1] #reverse x_temp
        
        #mixed samples vector 
        phi = np.append(y,x)
        v = np.dot(wtheta, phi)
        y_nl = np.tanh(v) #activation function
        
        #distance between the real and estimated values
        error = desired_data[k] - y_nl 
        
        #the atualization of the weights regards the activation function derivative 
        wtheta_out = wtheta + lr*error*cth*(ath - y_nl)*(ath + y_nl)*phi
        wtheta = wtheta_out
        
        #atualization of the y vector
        y_input = np.append(y_input,y_nl)
    
    
    y_out = y_input[winy:]
    
    return wtheta_out, y_out


def modified_perceptron(desired_data, input_data, lr):
    
    #signal parameters and generation
    wind = 3
    winx = 2
    
    xm = np.zeros(winx) 
    x_data = np.append(xm, input_data)
 
    winy = 3
    y_input = np.zeros(winy)
    
    #weights initialization
    a = np.zeros(wind)
    b = np.zeros(wind)
    wtheta = np.append(a,b)
   
    for k in range(0,len(input_data)):

        y_temp = y_input[k:wind+k]
        y = y_temp[::-1] #reverse y_temp
        
        x_temp = x_data[k:wind+k]
        x = x_temp[::-1] #reverse x_temp
        
        #mixed samples vector 
        phi = np.append(y,x)
        v = np.dot(wtheta, phi)
        y_nl = np.tanh(v) #activation function
        
        #distance between the real and estimated values
        error = s_data[k] - y_nl 
        
        #atualization of the weights
        wtheta_out = wtheta + lr*phi*error
        wtheta = wtheta_out
        
        #generation of y vector
        y_input = np.append(y_input,y_nl)
    
    y_out = y_input[winy:]
    
    return wtheta_out, y_out


def EQM(desired_data, y_data):
    
    esignal = []
    
    for k in range(0,len(y_data)):
        esample = (y_data[k] - desired_data[k])**2
        esignal.append(esample)

        eqm_temp = np.sum(esignal)
        eqm = eqm_temp/len(y_data)

    return eqm


# training
wtheta_mp, y_mp = modified_perceptron(s_data, x_data, 0.02)
wtheta_afmp, y_afmp = activation_function_derivative_modified_perceptron(s_data, x_data, 0.02)


#eqm
eqm_mp = EQM(s_data, y_mp)
eqm_afmp = EQM(s_data, y_afmp)


#figure
fig1 = plt.figure(figsize=(8.0, 6.0))
ax = fig1.gca()

plt.plot(n_samples[9900:10000], s_data[9900:10000], color='black', label="s(n)")
plt.plot(n_samples[9900:10000], y_mp[9900:10000], color='blue', label="y(n)")
plt.plot(n_samples[9900:10000], y_afmp[9900:10000], color='red', label="y_af(n)")

plt.xlabel('n')
plt.ylabel('s(n), yp(n), yafp(n)')
plt.legend(loc='upper right')

plt.grid()
plt.show()

