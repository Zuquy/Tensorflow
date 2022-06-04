#Extraido de: https://rukshanpramoditha.medium.com/how-to-use-tensorflow-adam-optimizer-to-solve-quadratic-equations-of-perfect-squares-16eb40cff1a7
import tensorflow as tf
import numpy as np

x= tf.Variable(0.0)
loss_fn = lambda: abs(x**2-6*x+9)
optimizer = tf.optimizers.Adam(learning_rate=0.1)
for i in range(10000):
  optimizer.minimize(loss_fn, x)
tf.print("The solution is:", x)
