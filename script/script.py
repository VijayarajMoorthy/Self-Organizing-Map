import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load the dataset (replace with the actual path)
df = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/SOM Proj/data/destinations.csv", encoding='ISO-8859-1')


# Preprocess the data: Select numeric columns, adjust depending on your dataset structure
data = df.select_dtypes(include=[float, int]).values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Initialize and train the Self-Organizing Map (SOM)
som_x, som_y = 10, 10  # SOM grid size
som = MiniSom(x=som_x, y=som_y, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Initialize weights
som.random_weights_init(data_scaled)

# Train the SOM (number of iterations can be adjusted)
som.train_random(data_scaled, 1000)

# Visualize the result
plt.figure(figsize=(10, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plot distance map
plt.colorbar()

# Optionally plot markers to show data points on the SOM grid
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(data_scaled):
    w = som.winner(x)  # Get the winning neuron
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[i % len(markers)], markerfacecolor='None', markeredgecolor=colors[i % len(colors)],  markersize=10, markeredgewidth=2)

# Show the plot
plt.show()
