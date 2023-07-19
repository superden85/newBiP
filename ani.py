import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np

saving_dir = 'F:\Documents\cours_mva\stages\Padova\july_work\meeting_21_7' 
n = 22338314 // 2

#open the file epochs_data.npy in the current directory
epochs_data = np.load('npy_files/epochs_data.npy', allow_pickle=True)
outer_gradients = epochs_data[0][-2]
masks = epochs_data[0][-1]

vectors = [mask[:100] for mask in masks]
fig, ax = plt.subplots()

# Initialize a scatter object for the plot
scatter = ax.scatter(range(len(vectors[0])), vectors[0])

# Define an update function for the animation
def update(i):
    scatter.set_offsets(np.c_[range(len(vectors[i])), vectors[i]])  # update the data of the scatter object
    return scatter,

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=len(vectors), interval=200)

path = os.path.join(saving_dir, 'animation.gif')
ani.save(path, writer='ffmpeg', fps=5)

plt.show()

print([mask[0] for mask in masks])