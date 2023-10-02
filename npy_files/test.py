import matplotlib.pyplot as plt

# Enable LaTeX text rendering
plt.rc('text', usetex=True)

# Data
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

# Create a plot
plt.plot(x, y)

# Set LaTeX labels
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.title(r'$\frac{\alpha}{\beta}$')

# Save the plot with LaTeX fonts
plt.savefig('latex_plot.png')
plt.show()