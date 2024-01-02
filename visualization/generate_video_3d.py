from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

label = 'T0.71_N7_RHO0.844'

# Load data for particles, energy, and temperature
data = pd.read_csv(f'../output_files/{label}_positions_data.txt', sep=' ', names=['particle', 'time', 'x', 'y', 'z'])
energy_data = pd.read_csv(f'../output_files/{label}_energy_data.txt', sep=' ', names=['time', 'K', 'V'])
temperature_data = pd.read_csv(f'../output_files/{label}_temperature_data.txt', sep=' ', names=['time', 'temperature'])

# Determine unique particles
unique_particles = data['particle'].unique()

# Assign a random radius and color to each particle
radius_dict = {particle: np.random.uniform(1, 5) for particle in unique_particles}  
color_dict = {particle: np.random.choice(['red', 'green', 'blue', 'purple']) for particle in unique_particles}

# Map these properties back to the DataFrame
#data['radius'] = data['particle'].map(radius_dict)
data['radius'] = 0.15
data['color'] = data['particle'].map(color_dict)

# Function to draw a sphere
def draw_sphere(ax, center, color, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.3)

# Set up the figure and GridSpec
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

# Create the subplots
ax1 = fig.add_subplot(gs[:, 0], projection='3d')  # 3D subplot for molecular movement
ax2 = fig.add_subplot(gs[0, 1])                    # Subplot for energy
ax3 = fig.add_subplot(gs[1, 1])                    # Subplot for temperature

# Initialize the 3D scatter plot
scat = ax1.scatter(data['x'], data['y'], data['z'])

# Line plot for energy
line1, = ax2.plot([], [], lw=2)
ax2.set_xlim(energy_data['time'].min(), energy_data['time'].max())
ax2.set_ylim(energy_data['V'].min(), energy_data['V'].max())
ax2.set_xlabel('Time')
ax2.set_ylabel('Energy')

# Line plot for temperature
line2, = ax3.plot([], [], lw=2)
ax3.set_xlim(temperature_data['time'].min(), temperature_data['time'].max())
ax3.set_ylim(temperature_data['temperature'].min(), temperature_data['temperature'].max())
ax3.set_xlabel('Time')
ax3.set_ylabel('Temperature')

# Initialize plots
line1, = ax2.plot([], [], lw=2)
line2, = ax3.plot([], [], lw=2)

# Update function for animation
def update(frame):
    ax1.clear()
    ax1.set_xlim([data['x'].min(), data['x'].max()])
    ax1.set_ylim([data['y'].min(), data['y'].max()])
    ax1.set_zlim([data['z'].min(), data['z'].max()])

    current_data = data[data['time'] == frame]
    for _, row in current_data.iterrows():
        draw_sphere(ax1, (row['x'], row['y'], row['z']), row['color'], row['radius'])  # Adjust radius as needed

    # Update energy plot
    current_energy = energy_data[energy_data['time'] <= frame]
    line1.set_data(current_energy['time'], current_energy['V'])

    # Update temperature plot
    current_temperature = temperature_data[temperature_data['time'] <= frame]
    line2.set_data(current_temperature['time'], current_temperature['temperature'])

    return ax1, line1, line2

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=data['time'].unique(), interval=50, blit=False)

# Save the animation
ani.save(f'../output_files/{label}_particle_motion_3D.mp4', writer='ffmpeg')

# Uncomment to display the plot in a Python script
# plt.show()
