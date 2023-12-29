import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data for particles and energy
data = pd.read_csv('../output_files/positions_data.txt', sep=' ', names=['particle', 'time', 'x', 'y'])
energy_data = pd.read_csv('../output_files/energy_data.txt', sep=' ', names=['time', 'energy'])
temperature_data = pd.read_csv('../output_files/temperature_data.txt', sep=' ', names=['time', 'temperature'])

np.random.seed(0)

# Determine unique particles
unique_particles = data['particle'].unique()

# Assign a random radius and color to each particle
radius_dict = {particle: np.random.uniform(1, 5) for particle in unique_particles}  
color_dict = {particle: np.random.choice(['red', 'green', 'blue', 'purple']) for particle in unique_particles}

# Map these properties back to the DataFrame
#data['radius'] = data['particle'].map(radius_dict)
data['radius'] = 1
data['color'] = data['particle'].map(color_dict)

# Prepare the subplots
fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# Scatter plot for molecular movement
scat = ax1.scatter(data['x'], data['y'])

# Line plot for energy
line1, = ax2.plot([], [], lw=2)
ax2.set_xlim(energy_data['time'].min(), energy_data['time'].max())
ax2.set_ylim(energy_data['energy'].min(), energy_data['energy'].max())
ax2.set_xlabel('Time')
ax2.set_ylabel('Energy')

# Line plot for temperature
line2, = ax3.plot([], [], lw=2)
ax3.set_xlim(temperature_data['time'].min(), temperature_data['time'].max())
ax3.set_ylim(temperature_data['temperature'].min(), temperature_data['temperature'].max())
ax3.set_xlabel('Time')
ax3.set_ylabel('Temperature')

# Initialization function for plots
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Update function for animation
def update(frame):
    # Update molecular movement
    current_data = data[data['time'] == frame]
    scat.set_offsets(current_data[['x', 'y']])

    # Update sizes based on the radius
    scat.set_sizes(current_data['radius']**2 * 100)  # Adjust 100 as needed

    # Update colors
    scat.set_color(current_data['color'])

    # Set transparency 
    scat.set_alpha(0.3)  

    # Update energy plot
    current_energy = energy_data[energy_data['time'] <= frame]
    line1.set_data(current_energy['time'], current_energy['energy'])

    # Update temperature plot
    current_temperature = temperature_data[temperature_data['time'] <= frame]
    line2.set_data(current_temperature['time'], current_temperature['temperature'])
    
    return scat, line1, line2

# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, frames=data['time'].unique(), interval=50, blit=True)

# Save the animation
ani.save('../output_files/particle_motion.mp4', writer='ffmpeg')
