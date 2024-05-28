# -*- coding: utf-8 -*-
"""
Simple generative AI using Gaussian Mixture Model

@author: Tomas Arzola Röber
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture

# Load the digits dataset
digits = load_digits()
plt.gray()

# Display an original image from the dataset
plt.matshow(digits.images[6])
plt.show()

# Separate the dataset into features and target
X = digits['data']
y = digits['target']

# Create an instance of the GaussianMixture class and train the model
gm = GaussianMixture(n_components=100, random_state=42)
gm.fit(X)

# Generate 12 new images of numbers
samples, _ = gm.sample(n_samples=12)

# Plot the 12 generated images in a 3x4 grid
fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    ax.matshow(samples[i].reshape(8, 8))
    ax.axis('off')

plt.show()
