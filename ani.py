import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create some data
data = np.random.rand(10, 100)

fig, ax = plt.subplots()

# Initialize a line object for the plot
line, = ax.plot(data[0, :])

# Define an update function for the animation
def update(i):
    line.set_ydata(data[i, :])  # update the data of the line object
    return line,

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=range(10), interval=200)

plt.show()
