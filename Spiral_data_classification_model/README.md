# Neural Network Classification Model for Predicting Spiral Data

This repository contains a neural network classification model developed from scratch, guided by the book "Neural Networks from Scratch." The model is designed to classify spiral data belonging to three different classes. Unlike early regression examples, this implementation includes a model class, making training and forward/backward passes more streamlined.

## **Project Overview**
- **Building Blocks**: The neural network is built using the following components:
  - Dense layers
  - Activation functions (ReLU and Softmax)
  - Loss functions (Categorical Cross Entropy)
  - Optimizers (Adam Optimizer)
- **Data Generation**: The dataset consists of randomly generated spiral data for three classes. More details about the data generation process can be found in the code and in the book.
- **Model Class**: The `network.py` file includes a model class, which automates the training process, including forward and backward passes, updates, loss calculation, and accuracy evaluation.

## **Model Structure**
- The network is composed of one hidden  dense layers.
- **Activation Functions**: ReLU is used for hidden layers, while the final layer uses the Softmax activation function for classification.
- **Loss Calculation**: Categorical Cross Entropy is used as the loss function for this classification problem.
- **Optimization**: The Adam optimizer is employed to update the model parameters efficiently.

## **Files in This Repository**
- `network.py`: Contains the core classes for the neural network, including dense layers, activation functions, optimizers, loss functions, and the model class.
- `Model_def&train.py`: Contains the code for building, training, and evaluating the classification model.

## **Usage**
1. Clone the repository:
2. Explore `network.py` to understand the building blocks and model class.
3. Run `Model_def&train.py` to train the model on the spiral dataset.
4. Modify the code as needed to experiment with different configurations.

## **Conclusion**
This repository showcases a classification model for predicting spiral data from scratch. It serves as a hands-on example for understanding the fundamentals of neural networks, including data generation, manual training processes, and model evaluation. For more details, refer to the classification section of the book "Neural Networks from Scratch."

Feel free to explore the code, experiment with the model, and share your feedback!
