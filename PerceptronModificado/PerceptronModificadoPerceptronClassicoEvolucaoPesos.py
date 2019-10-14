import pandas as pd
import numpy as np
import random as rd
from scipy.signal import lfilter
import matplotlib
from matplotlib import pyplot as plt

#Time Series
n = 10000
n_samples = np.linspace(0,n-1,n)

white_noise = np.random.uniform(-0.9, 0.9, n) #uniform white noise with 10000 samples between -0.9 and 0.9
s_data = white_noise
u_data = np.arctan(s_data) #addition of nonlinearity in white noise
x_data = lfilter([1, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0.5], 1, u_data) #filtered input time serie

#Desired Time Serie Plot (White Noise between -1 and 1)
fig = plt.figure(figsize=(8.0, 6.0))
ax = fig.gca()
plt.plot(n_samples[0:99], s_data[0:99], color='black')
plt.xlabel('n')
plt.ylabel('s(n)')
plt.grid()
plt.show()

#Time Serie With Distortion
fig2 = plt.figure(figsize=(8.0, 6.0))
ax = fig2.gca()
plt.plot(n_samples[0:99], u_data[0:99], color='blue')
plt.xlabel('n')
plt.ylabel('u(n)')
plt.grid()
plt.show()

#Input Time Serie Plot (Filtered Time Serie)
fig3 = plt.figure(figsize=(8.0, 6.0))
ax = fig3.gca()
plt.plot(n_samples[0:99], x_data[0:99], color='red')
plt.xlabel('n')
plt.ylabel('x(n)')
plt.grid()
plt.show()

#perceptron classico sinal
def perceptron(x_input, s_data, bias, lr):
    wind = 3
    winx = 2
    xm = np.zeros(winx)

    x_data = np.append(xm, x_input)

    w = np.zeros(wind)
    w_temp = np.append(bias, w) #dim = 4

    y_out = []
    W_outc = [] #store the weights already updated

    #for k in range (0, len(x_input)-1):
    for k in range (0, len(x_input)):

        w = w_temp #dim = 4

        x_temp = x_data[k:wind+k] #dim = 3
        x_temp2 = x_temp[::-1] #reverse x_temp
        x_temp3 = np.append(1, x_temp2) #dim = 4

        #y_temp is the dot product between w_temp and x_temp
        y_temp = np.tanh(np.dot(w,x_temp3)) #scalar value
        y_out.append(y_temp)

        if y_temp == s_data[k]: #if y_sample is equal to the desired sample, if the class is correct
            w_out = w
        else:
            w_out = w + lr*(s_data[k] - y_temp)*x_temp3


        W_outc.append(w_out) #temporal evolution of the weights
        w_temp = w_out

    return w_out, y_out, W_outc

#perceptron modificado sinal
def PLR(s_data, x_input, lr):

    wind = 3
    winx = 2
    xm = np.zeros(winx)
    x_data = np.append(xm, x_input)

    winy = 3
    y_input = np.zeros(winy)

    a = np.zeros(wind)
    b = np.zeros(wind)

    wtheta = np.append(a,b)

    for k in range(0,len(x_input)):
    #for k in range(0,len(x_input[winx:])-1):

        y_temp = y_input[k:wind+k]
        y = y_temp[::-1] #reverse y_temp

        x_temp = x_data[k:wind+k]
        x = x_temp[::-1] #reverse x_temp

        phi = np.append(y,x) #vector
        y_n = np.dot(wtheta, phi) #scalar, is the value of the n sample of the y signal

        error = s_data[k] - y_n  #scalar
        wtheta_out = wtheta + lr*phi*error #vector, weights update

        wtheta = wtheta_out
        y_input = np.append(y_input,y_n)

        y_out = y_input[winy:]

    return wtheta_out, y_out

#perceptron modificado não linearidade tanh sinal
def PLR_NL(s_data, x_input, lr):

    wind = 3
    winx = 2
    xm = np.zeros(winx)
    x_data = np.append(xm, x_input)

    winy = 3
    y_input = np.zeros(winy)

    a = np.zeros(wind)
    b = np.zeros(wind)

    wtheta = np.append(a,b)
    W_outm = []

    for k in range(0,len(x_input)):
        y_temp = y_input[k:wind+k]
        y = y_temp[::-1] #reverse y_temp

        x_temp = x_data[k:wind+k]
        x = x_temp[::-1] #reverse x_temp

        phi = np.append(y,x)
        y_n = np.dot(wtheta, phi)
        y_nl = np.tanh(y_n)

        error = s_data[k] - y_nl
        wtheta_out = wtheta + lr*phi*error

        wtheta = wtheta_out
        y_input = np.append(y_input,y_nl)

        W_outm.append(wtheta_out) #temporal evolution of the weights

    y_out = y_input[winy:]

    return wtheta_out, y_out, W_outm

#Erro Quadratico Medio
def EQM(s_data, y_data):

    esignal = []

    for k in range(0,len(y_data)):
        esample = (y_data[k] - s_data[k])**2
        esignal.append(esample)

    eqm_temp = np.sum(esignal)
    eqm = eqm_temp/len(y_data)

    return eqm

#perceptron modificado:
wtheta_plr, y_plr, W_outplr = PLR_NL(s_data, x_data, 0.02)
eqm_plr = EQM(s_data, y_plr)

#perceptron classico
wtheta_perceptron, y_perceptron, W_outperceptron = perceptron(x_data, s_data, 0.5, 0.02)
eqm_perceptron = EQM(s_data, y_perceptron)

print(eqm_perceptron)
print(eqm_plr)

fig4 = plt.figure(figsize=(15.0, 5.0))
ax = fig4.gca()
plt.plot(n_samples[0:199], s_data[0:199], color='black', label= 's(n)')
plt.plot(n_samples[0:199], y_perceptron[0:199], color='red', label= 'y_c(n)')
plt.plot(n_samples[0:199], y_plr[0:199], color='blue', label= 'y_m(n)')
plt.xlabel('n')
#plt.ylabel('s(n), y_c(n), y_m(n)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

#temporal evolution of the weights
x = [] #time evolution
#Perceptron classico
w1c = []
w2c = []
w3c = []

#Perceptron modificado
w1m = []
w2m = []
w3m = []

for i in range(0,(len(W_outplr)-1)):
    x.append(i)

    Woutc_i = W_outperceptron[i] #matrices of each time step classic
    w1c.append(Woutc_i[1])
    w2c.append(Woutc_i[2])
    w3c.append(Woutc_i[3])

    Woutm_i = W_outplr[i] #matrices of each time step modified
    w1m.append(Woutm_i[1])
    w2m.append(Woutm_i[2])
    w3m.append(Woutm_i[3])


fig5 = plt.figure(figsize=(15.0, 5.0))
ax = fig5.gca()
plt.plot(x[0:1999],w1c[0:1999], color='red', label='w1_c(n)')
plt.plot(x[0:1999],w1m[0:1999], color='blue', label='w1_m(n)')
plt.xlabel('n')
#plt.ylabel('w1c(n), w1m(n)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

fig6 = plt.figure(figsize=(15.0, 5.0))
ax = fig6.gca()
plt.plot(x[0:1999],w2c[0:1999], color='red', label='w2_c(n)')
plt.plot(x[0:1999],w2m[0:1999], color='blue', label='w2_m(n)')
plt.xlabel('n')
#plt.ylabel('w2c(n), w2m(n)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

fig7 = plt.figure(figsize=(15.0, 5.0))
ax = fig7.gca()
plt.plot(x[0:1999],w3c[0:1999], color='red', label='w3_c(n)')
plt.plot(x[0:1999],w3m[0:1999], color='blue', label='w3_m(n)')
plt.xlabel('n')
#plt.ylabel('w3c(n), w3m(n)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

fig8 = plt.figure(figsize=(15.0, 5.0))
ax = fig8.gca()
plt.plot(x[0:1999],w1c[0:1999], color='green', label='w1_c(n)')
plt.plot(x[0:1999],w1m[0:1999], color='darkgreen', label='w1_m(n)')
plt.plot(x[0:1999],w2c[0:1999], color='blue', label='w2_c(n)')
plt.plot(x[0:1999],w2m[0:1999], color='navy', label='w2_m(n)')
plt.plot(x[0:1999],w3c[0:1999], color='red', label='w3_c(n)')
plt.plot(x[0:1999],w3m[0:1999], color='darkred', label='w3_m(n)')
plt.xlabel('n')
#plt.ylabel('w3c(n), w3m(n)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

print('Aleluia')
