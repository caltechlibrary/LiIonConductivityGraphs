import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

directory = r"C:\Users\mchaf\Downloads"
filename = "LiNaFeS2_cycle1"
filepath = os.path.join(directory, filename + ".txt")
with open(filepath, 'r') as file:
    data = file.readlines()

# Parse data into lists of x and y values
x_values = []
y_values = []
for line in data:
    y, x = map(float, line.split())
    x_values.append(x)
    y_values.append(y)

# Calculate average step size
average_dx = np.mean(np.diff(x_values))

# Calculate first derivative using forward finite difference
first_derivative = []
for i in range(len(y_values) - 1):
    dx = x_values[i+1] - x_values[i]
    dy = y_values[i+1] - y_values[i]
    if dx != 0:
        derivative = dy / dx
    else:
        # Use the average step size as dx
        derivative = dy / average_dx
    first_derivative.append(derivative)

x_derivative = x_values[:-1]

# Apply Savitzky-Golay filter to smooth the y-values
window_length = 5  # Adjust the window length as needed
poly_order = 2     # Adjust the polynomial order as needed
y_smooth = savgol_filter(y_values, window_length, poly_order)

# Now compute the derivative using the smoothed data
# Calculate first derivative using forward finite difference
smoothed_derivative = []
for i in range(len(y_smooth) - 1):
    dx = x_values[i+1] - x_values[i]
    dy = y_smooth[i+1] - y_smooth[i]
    if dx != 0:
        derivative = dy / dx
    else:
        # Use the average step size as dx
        derivative = dy / average_dx
    smoothed_derivative.append(derivative)

output_filename = os.path.join(directory, filename + "_firstderivative.txt")
with open(output_filename, 'w') as file:
    for x, derivative in zip(x_derivative, first_derivative):
        file.write(f"{x}\t{derivative}\n")