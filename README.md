# MNIST Digit Classification using KSOM and k-NN

This project implements a hybrid approach to digit classification on the **MNIST** dataset using a combination of 
**Kohonen Self-Organizing Map (KSOM)** for unsupervised feature learning and **k-Nearest Neighbors (k-NN)** 
for classification. The entire pipeline is built using **NumPy**, without relying on deep learning libraries, 
to gain a deeper understanding of how these algorithms work under the hood.

---

## 🚀 Project Features

- **Custom KSOM Implementation**  
  A 2D Kohonen Self-Organizing Map is used to learn spatial patterns from flattened MNIST digits. 
  It organizes high-dimensional data onto a 2D neuron grid using competitive learning and neighborhood-based updates.

- **Digit Mapping Visualization**  
  After training, the grid displays which neurons have learned which digits, 
  giving insights into how different digits are clustered spatially.

- **Feature Selection from SOM Weights**  
  Feature importance is derived from neuron weight norms. 
  The top 100 most relevant features are selected for the next classification stage.

- **k-NN Classification (k = 3)**  
  A simple yet effective supervised classifier is used after feature reduction 
  to predict digits from the test set.

- **Performance Evaluation**  
  - KSOM grid training achieves **~70–80% unsupervised label clustering accuracy** over 25 epochs.  
  - Final k-NN model reaches around **92% accuracy** on the test subset using reduced features.

- **Digit Heatmap Visualization**  
  A randomly selected digit from the test set is overlaid with a synthetic importance heatmap 
  to illustrate pixel-level relevance.

---

## 📈 Results Snapshot

Epoch-wise output:
    Epoch 1/25 - Accuracy: 76.32% - Loss: 3.6541
    ...
    Epoch 25/25 - Accuracy: 84.27% - Loss: 2.0385

Final test accuracy:
    Model accuracy: 91.80%

---

## 🛠 Tools & Technologies

- Python
- NumPy
- Matplotlib
- TensorFlow/Keras (only used to load MNIST dataset)
