import matplotlib.pyplot as plt
import numpy as np
x_axis_value = np.array([4, 9, 6, 7, 12, -13, -21, 19, 4, 11,-22, 18, 6])
y_axis_value = np.array([90, -80, 89, 88, 101, 82, 102, 80, -91, 76, 77, 88, 66])
#plot basic scatter value in python  
plt.scatter(x_axis_value, y_axis_value)
x_axis_value = np.array([5, 12, 8, 11, 16, 8, 2, 18, 7, 3, 21, 14, 17, 24, 12])
y_axis_value = np.array([101, 106, 85, 105, 90, 98, 90, 95, 94, 99, 78, 118, 99, 88, 85])
#plot basic scatter value in python  
plt.scatter(x_axis_value, y_axis_value)
plt.show()
