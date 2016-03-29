import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import compiler

# Define 'a' and 'b' and number of grid points (the more the better)

a = float(raw_input("Please enter your value of 'a': "))
b = float(raw_input("Please enter your value of 'b': "))
g = float(raw_input("Please enter the number of gridpoints: "))
x = np.linspace(a,b,num=g)

# Define your 'y' function

f = raw_input("Please enter your equation 'y' in Python format: ")
y = eval(f)

# Integration section

x1 = simps(x*y,x)
d = simps(y,x)
y1 = simps(y*y,x)
x_bar = x1/d
y_bar = 0.5*y1/d
print "X_bar = %s." % x_bar
print "Y_bar = %s." % y_bar
# Plotting curve

plt.figure()
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of your curve')
plt.savefig('Curve.png')	