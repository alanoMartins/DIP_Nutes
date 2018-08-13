from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def function(x, y):
    return (x*x)-(y*y)

def derivate_of_function_x(x): #dx
    return 2*x
def derivate_of_function_y(y): #dy
    return -2*y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = 38
y = -18

lr = 0.1

for i in range(10):
    # print(function(x, y))
    print(x, '-', y)
    x += derivate_of_function_x(x) * -1 * lr
    y += derivate_of_function_y(y) * -1 * lr
    ax.scatter(x, y, function(x, y), c="r")

print(x, y, function(x,y))
ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Z Label')

plt.show()