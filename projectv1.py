import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv'
gps_data = pd.read_csv(file_path)

# Rename columns for easier access
gps_data.columns = ["Timestamp", "Longitude", "Latitude", "Speed", "Status"]

# Convert lat/lon to local coordinates (meters)
origin_lat = np.radians(gps_data["Latitude"].iloc[0])
origin_lon = np.radians(gps_data["Longitude"].iloc[0])
earth_radius = 6378137.0

def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = earth_radius * (lon - origin_lon) * np.cos(origin_lat)
    y = earth_radius * (lat - origin_lat)
    return x, y

gps_data["X"], gps_data["Y"] = latlon_to_xy(gps_data["Latitude"], gps_data["Longitude"], origin_lat, origin_lon)

# Constants for Stanley control
constant_speed = 5 * 0.44704  # Constant speed (5 mph) in meters per second
time_step = 0.1               # Time step in seconds
kp_stanley = 1.0              # Gain for Stanley control
wheelbase = 2.5               # Typical wheelbase length for a small vehicle in meters

# Initialize variables
positions_stanley = []
heading = 0  # Initial heading in radians

# Simulation loop with Stanley control for lateral guidance
for i in range(1, len(gps_data)):
    # Target position and current position
    target_position = gps_data.loc[i, ["X", "Y"]].values
    current_position = gps_data.loc[i - 1, ["X", "Y"]].values
    
    # Cross-track error for Stanley control
    position_error = target_position - current_position
    cross_track_error = np.linalg.norm(position_error)
    
    # Path angle and heading error
    path_angle = np.arctan2(position_error[1], position_error[0])
    heading_error = path_angle - heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize to [-pi, pi]
    
    # Stanley control law for steering angle
    steering_angle = heading_error + np.arctan2(kp_stanley * cross_track_error, constant_speed)
    
    # Update heading based on steering
    heading += steering_angle * time_step  # Update heading with steering adjustment

    # Update vehicle's position
    new_position = current_position + np.array([np.cos(heading), np.sin(heading)]) * constant_speed * time_step
    positions_stanley.append(new_position)

# Convert Stanley controlled positions to a DataFrame for plotting
simulated_path_stanley = pd.DataFrame(positions_stanley, columns=["X", "Y"])

# Plotting the actual GPS path and the simulated Stanley-controlled path
plt.figure(figsize=(8, 12))
plt.plot(gps_data["X"], gps_data["Y"], label="Actual GPS Path", linestyle="-", color="blue")
plt.plot(simulated_path_stanley["X"], simulated_path_stanley["Y"], label="Simulated Path (Stanley Control)", linestyle="--", color="green")

# Mark the starting and stopping points
plt.scatter(gps_data["X"].iloc[0], gps_data["Y"].iloc[0], color="orange", label="Starting point", marker="*", s=100)
plt.scatter(gps_data["X"].iloc[-1], gps_data["Y"].iloc[-1], color="red", label="Stopping point", marker="^", s=100)

# Labels and legend
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("Actual GPS Path vs. Simulated Path with Stanley Control")
plt.legend()
plt.grid(True)
plt.show()
