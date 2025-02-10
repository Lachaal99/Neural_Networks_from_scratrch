# Fashion MNIST Classification Model with Save/Load Functionality

This repository contains a neural network classification model trained on the Fashion MNIST dataset, guided by the book "Neural Networks from Scratch." Building upon the core project (which can be found in the Other Fashion_mnist folder), this version introduces additional functionality to save and load the entire model, including its architecture, parameters, and optimizer state, rather than just the hyperparameters.

## **Project Overview**
 - Dataset: The Fashion MNIST dataset is used for training and evaluating the model. It consists of grayscale images of 10 different clothing categories.

- Core Workflow: The workflow remains the same as the original project, with the following components:

    - Dense layers

    - Activation functions (ReLU and Softmax)

    - Loss function (Categorical Cross Entropy)

    - Optimizer (Adam Optimizer)

    - Accuracy calculation

    New Feature: Added methods to save and load the entire model, including its architecture, weights, biases, and optimizer state.

## **Model Structure**

- The network includes multiple dense layers to process the image data.

- Activation Functions: ReLU is used for hidden layers, and Softmax is used for the output layer to handle the classification.

- Loss Calculation: Categorical Cross Entropy is used to calculate the loss for this classification problem.

- Optimization: The Adam optimizer is employed for updating model parameters.

## ** New Functionality**
- Save Model: The model can now be saved to a file, including:

- Architecture (layer configurations)

- Weights and biases

- Optimizer state 

- Load Model: The saved model can be loaded back into memory, allowing for seamless continuation of training or inference without reinitializing the model.

## **Files in This Repository**
- network_fully.py: Contains the neural network building blocks, the model class, and the new save/load methods.

- model.py: Handles preprocessing, model setup, training, and evaluation. It also includes functionality to save th model.

- Fashion_mnist.model: A saved model.

- Model_load&prediction.py : loading the saved model and making inference. 

## **Conclusion**
This project extends the original Fashion MNIST classification model by adding the ability to save and load the entire model, making it easier to resume training or deploy the model for inference. It serves as a practical example of how to implement model persistence in neural networks. For more details, refer to the book's Fashion MNIST classification section and explore the code in the Other Fashion_mnist folder.

Feel free to explore the code, run experiments, and share your feedback!