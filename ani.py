import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create some data
data = np.random.rand(10, 100)

fig, ax = plt.subplots()

# Initialize a scatter object for the plot
scatter = ax.scatter(range(len(vectors[0])), vectors[0])

# Define an update function for the animation
def update(i):
    scatter.set_offsets(np.c_[range(len(vectors[i])), vectors[i]])  # update the data of the scatter object
    return scatter,

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=len(vectors), interval=200)

plt.show()
