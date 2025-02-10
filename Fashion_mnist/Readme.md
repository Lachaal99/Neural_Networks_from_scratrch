# Neural Network Classification Project Using the Fashion MNIST Dataset

This repository contains a neural network classification model trained on the Fashion MNIST dataset. The example is guided by the book "Neural Networks from Scratch." After developing all the core components of a neural network (dense layers, activation functions, loss calculation, optimization, and accuracy calculation), these components were integrated into a model class to automate the entire process, from forward and backward passes to parameter updates.

## **Project Overview**
- **Dataset**: The Fashion MNIST dataset is used for training and evaluating the model. It consists of grayscale images of 10 different clothing categories.
- **Building Blocks**: The neural network components include:
  - Dense layers
  - Activation functions (ReLU and Softmax)
  - Loss function (Categorical Cross Entropy)
  - Optimizer (Adam Optimizer)
  - Accuracy calculation
- **Model Class**: The `network_fully.py` file contains the core implementation of the neural network, including the model class with an added evaluation method for testing the model.

## **Model Structure**
- The network includes multiple dense layers to process the image data.
- **Activation Functions**: ReLU is used for hidden layers, and Softmax is used for the output layer to handle the classification.
- **Loss Calculation**: Categorical Cross Entropy is used to calculate the loss for this classification problem.
- **Optimization**: The Adam optimizer is employed for updating model parameters.

## **Files in This Repository**
- `network_fully.py`: Contains the neural network building blocks and the model class, including methods for training and evaluation.
- `model.py`: Handles the preprocessing operations and model setup, including data preparation, model building, and training.

## **Usage**
1. Clone the repository:
2. Run `model.py` to preprocess the data, build the model, and train it on the Fashion MNIST dataset.
3. Use the evaluation method to test the model and check its performance.
4. Modify the code to experiment with different configurations and hyperparameters.

## **Conclusion**
This project demonstrates a neural network classification model applied to the Fashion MNIST dataset. By integrating all the core neural network components into a model class, it automates the training and evaluation processes, offering a practical example for understanding neural networks. For more details, refer to the book's Fashion MNIST classification section.

Feel free to explore the code, run experiments, and don't forget to download the data into the repository after cloning!

