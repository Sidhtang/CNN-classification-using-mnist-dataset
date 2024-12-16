# CNN-classification-using-mnist-dataset
Data Preprocessing and Splitting
Used torchvision.datasets.MNIST to load the dataset
Applied normalization with mean and standard deviation specific to MNIST
Split dataset into:
Training set (80%)
Validation set (20%)
Test set (separate MNIST test dataset
Neural Network Architecture
The MNISTClassifier is a Convolutional Neural Network with:
Two convolutional layers
Batch normalization layers
Dropout layers
Fully connected layers
Batch Normalization and Dropout Layers
Batch Normalization
Purpose: Normalizes the inputs of each layer to have zero mean and unit variance
Benefits:
Stabilizes and accelerates training
Reduces internal covariate shift
Allows higher learning rates
Acts as a mild regularizer
In my  model, i  added nn.BatchNorm2d() after each convolutional layer to:
Normalize feature maps
Improve gradient flow
Speed up training convergence
Dropout Layers
Purpose: Randomly "drops out" (sets to zero) a proportion of neurons during training
Benefits:
Prevents overfitting
Introduces regularization
Encourages robust feature learning
Simulates an ensemble of neural networks
In my  model, i used dropout with different rates:
0.25 after convolutional layers
0.5 in the fully connected layer
Training Strategy
Used Adam optimizer with learning rate of 0.001
CrossEntropyLoss for multi-class classification
10 training epochs
Tracked both training and validation losses
Visualization
Created a function plot_losses() to visualize training and validation loss curves, helping understand model's learning progress.
Performance Evaluation
Calculated test set accuracy
Prints per-epoch training and validation losses
