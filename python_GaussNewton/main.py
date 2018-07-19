# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:48:26 2018

@author: ses516
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
# Dense Matrix Dimensions

threshold, alpha = 1e-8, 0.01
A, B, C, D = 0.5, 1.2, 0.3, 0.6
noise_lvl = 0.1
x = np.linspace(0,10,1001)
y = A * np.sin(B * x + C) + D + np.random.normal(scale = noise_lvl, size = x.shape)
#plt.plot(y)

beta = np.random.rand(4,1)
#beta[1] = 49.8
beta_tf = tf.Variable(beta,tf.double)
x_tf = tf.constant(x.reshape([1001,1]))
model = beta_tf[0] * tf.sin(beta_tf[1] * x_tf + beta_tf[2]) + beta_tf[3]
y_tf = tf.constant(y.reshape([1001,1]))
#loss = tf.reduce_sum(tf.square(y_tf - model))
#J = tf.test.compute_gradient(beta_tf,(4,1),model,(1001,1))
#J = tf.gradients(model,beta_tf)
diff, i, temp = 1, 0, 1
loss = np.zeros([1,1])
#beta_tf = beta_tf + alpha * tf.transpose(delta)

def del_beta(beta_tf, x_tf, y_tf):
    j1 = tf.sin(beta_tf[1] * x_tf + beta_tf[2])
    j2 = beta_tf[0] * x_tf * tf.cos(beta_tf[1] * x_tf + beta_tf[2])
    j3 = beta_tf[0] * tf.cos(beta_tf[1] * x_tf + beta_tf[2])
    j4 = tf.cast(tf.ones(x.shape), tf.double)
    j1 = tf.reshape(j1, [1001,1]); j2 = tf.reshape(j2, [1001,1])
    j3 = tf.reshape(j3, [1001,1]); j4 = tf.reshape(j4, [1001,1])
    J = tf.concat([j1, j2, j3, j4], 1)
    f = beta_tf[0] * tf.sin(beta_tf[1] * x_tf + beta_tf[2]) + beta_tf[3]
    r = tf.reshape(tf.transpose(y_tf - f),[1001,1])
    d = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(J),J)),tf.matmul(tf.transpose(J),r))
    loss = tf.reshape(tf.reduce_sum(tf.square(r)), [1,1]) 
    return [d, loss]

delta, temp = del_beta(beta_tf, x_tf, y_tf)
update = tf.assign(beta_tf,beta_tf + alpha * delta)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

while diff > threshold:
    i += 1
#    delt, temp = sess.run(del_beta(beta_tf, x_tf, y_tf))
#    sess.run(update,feed_dict = {delta: delt.T})
    sess.run(update)
    loss = np.append(loss,sess.run(temp), axis = 0)
    if i == 1:
        diff = abs(loss[-1])
    else:
        diff = abs(loss[-1] - loss[-2])
        
beta_est = sess.run(beta_tf)
plt.figure(1); plt.plot(x, y)
plt.figure(1); plt.plot(x, beta_est[0] * np.sin(beta_est[1] * x + beta_est[2]) + beta_est[3])
plt.show()
        
plt.figure(2); plt.plot(loss)
plt.show()

print("loss function value: ", loss[-1])
print("Presumed Parameters: ", [[A, B, C, D]])
print("Estimated Parameters: ", beta_est.T)

    























