import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset and split into train & test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the images and normalize pixel values
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Take a random 25% sample of training data
train_indices = np.random.choice(x_train.shape[0], x_train.shape[0] // 4, replace=False)
test_indices = np.random.choice(x_test.shape[0], x_test.shape[0] // 4, replace=False)
x_train, y_train = x_train[train_indices], y_train[train_indices]
x_test, y_test = x_test[test_indices], y_test[test_indices]

# KSOM implementation
grid_size = (10, 10) # size of grid for neurons
input_dim = x_train.shape[1] #dimntionality of input data
learning_rate = 0.1 # Initial Learning rate
sigma = 1.0 # Initial Standard Deviation
weights = np.random.rand(grid_size[0], grid_size[1], input_dim) # Initialize random weights
grid_map = np.full(grid_size, None) # initialize grid mapping

# KSOM training
num_iterations = 25 # number of training epochs

for iteration in range(num_iterations):
    grid_map = np.full(grid_size, None)  # Reset grid mapping for each epoch
    total_loss = 0 # initialize total loss - Sum of distances btw i/p vectors and their BMUs
    correct_mappings = 0 # initialize correct mapping- counter to check how many inputs are correctly mapped to their BMUs

    # Find Best Matching Unit(BMU)
    for idx, input_vector in enumerate(x_train):
        distances = np.linalg.norm(weights - input_vector, axis=2) # calculate distances between i/p vector and all weights
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape) # Find Best Matching Unit
        total_loss += distances[bmu_index]  # Add the distance to the BMU as loss

        # update weights of neurons within the neighborhood
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_index)) #neighborhood distance
                theta = np.exp(-distance_to_bmu / (2 * (sigma ** 2))) #neighborhood function
                weights[i, j] += learning_rate * theta * (input_vector - weights[i, j]) #update weights

        # maps i/p label to BMU on the grid map
        if grid_map[bmu_index] is None:
            grid_map[bmu_index] = [y_train[idx]]
        else:
            grid_map[bmu_index].append(y_train[idx])

    # Calculate accuracy
    for bmu in np.ndindex(grid_size): # Iterate over all BMU positions
        if grid_map[bmu] is not None:
            label_counts = np.bincount(grid_map[bmu])  # Count occurences of each label for BMU
            dominant_label = np.argmax(label_counts)  # Find the dominant label
            correct_mappings += label_counts[dominant_label] # Add dominant counts

    accuracy = correct_mappings / len(y_train) #calculate the accuracy
    avg_loss = total_loss / len(x_train) # calculate average loss for each epoch

    # Print metrics for current epoch
    print(f"Epoch {iteration + 1}/{num_iterations} - Accuracy: {accuracy * 100:.2f}% - Loss: {avg_loss:.4f}")


# Visualize KSOM grid
plt.figure(figsize=(10, 10))
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if grid_map[i, j] is not None:
            label_counts = np.bincount(grid_map[i, j]) # count labels at each grid point
            dominant_label = np.argmax(label_counts) # find most common label
            plt.text(j, i, str(dominant_label), ha='center', va='center', fontsize=12)
plt.xlim(-0.5, grid_size[1] - 0.5) # Set x-axis limit
plt.ylim(-0.5, grid_size[0] - 0.5) # Set y-axis limit
plt.gca().invert_yaxis() # Gets current axes and Inverts y-axis for correct orientation
plt.title("KSOM Digit Mapping") #Plots title
plt.show()

# Feature selection
importance = np.linalg.norm(weights, axis=(0, 1))  # Calculate features importance
num_features = 100 # number of features to select(top 100 here)
selected_indices = np.argsort(importance)[-num_features:]  # Indices of most important features

# Reduce the dataset to the selected features
reduced_x_train = x_train[:, selected_indices] # train data with selected features
reduced_x_test = x_test[:, selected_indices] # test data with selected features

# k-NN implementation
k = 3 # number of neighbors for k-NN is 3
train_data = reduced_x_train # K-NN training data
train_labels = y_train # K-NN raining labels

predictions = []
for test_point in reduced_x_test:
    distances = np.linalg.norm(train_data - test_point, axis=1)  # Calculate distances to neighbors
    nearest_indices = np.argsort(distances)[:k] # Find indices of k nearest neighbors
    nearest_labels = train_labels[nearest_indices] # labels of nearest neighbors
    predictions.append(np.bincount(nearest_labels).argmax()) # predict most common label
predictions = np.array(predictions) #convert predictions to numpy array

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

random_idx = np.random.randint(0, x_test.shape[0])  # Select a random index from test set
random_image = x_test[random_idx].reshape(28, 28)  # Reshape test image to its original image (28x28)
true_label = y_test[random_idx]  # Gives actual label for selected image

# Generate a random "importance map" of the same size as the original image
importance_map = np.abs(random_image * np.random.rand(28, 28))

# Normalize the importance map for better visualization
importance_map = (importance_map - np.min(importance_map)) / \
                 (np.max(importance_map) - np.min(importance_map))

# Plot the original image and its heatmap
plt.figure(figsize=(10, 5))  # figsize : display 2 subplots side by side

# Display original MNIST image in grayscale
plt.subplot(1, 2, 1)
plt.imshow(random_image, cmap='gray') # display original image
plt.title(f"True Label: {true_label}") # Title for label
plt.axis('off') # hide axes

# Heatmap overlaid on the original image
plt.subplot(1, 2, 2)
plt.imshow(random_image, cmap='gray') # display grayscale image
plt.imshow(importance_map, cmap='hot', alpha=0.6)  # Overlay importance map with transparency
plt.title("Digit Heatmap") # Title for heatmap
plt.colorbar() # add color bar to interpret the heatmap values
plt.axis('off') # Hide axes

plt.tight_layout() # Adjust spacing between subplots(layout) for better readability
plt.show() # Display the final visualization