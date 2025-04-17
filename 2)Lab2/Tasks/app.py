# Scalar Step Function
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else 0

# Vectorized Step Function (TensorFlow)
def step_tf(x):
    return tf.cast(x >= 0, tf.int32)

# Test Input
inputs1 = [-3.0, -1.0, 0.0, 2.0, 5.0]
outputs1 = [step_function(x) for x in inputs1]
print("Step Function Output:", outputs1)

# TensorFlow Version
inputs1_tf = tf.constant(inputs1)
outputs1_tf = step_tf(inputs1_tf)
print("Step Function TF:", outputs1_tf.numpy())

# Visualization
x_vals = np.linspace(-5, 5, 200)
y_vals = [step_function(x) for x in x_vals]

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals, label="Step Function")
plt.title("Step Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()