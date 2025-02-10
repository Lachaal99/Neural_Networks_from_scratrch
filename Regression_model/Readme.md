## **Project Overview**
- **Building Blocks**: The following components were implemented to build and train the neural network:
  - Dense layers
  - Activation functions (ReLU and others)
  - Loss functions (Mean Squared Error for this regression task)
  - Optimizers (Adam Optimizer)
- **Model Purpose**: This model is designed to predict sine function values. It is a simple regression model that can be run and tested easily.

## **Model Structure**
- The network consists of fully connected dense layers.
- **Activation Functions**: ReLU is used for all layers except the output layer where we used a linear activation function.
- **Loss Calculation**: Mean Squared Error (MSE) is used as the loss function since this is a regression problem.
- **Optimization**: The Adam optimizer is used for updating the model parameters. Detailed specifications can be found in the code.

## **Files in This Repository**
- `network.py`: Contains the core classes for the neural network, including dense layers, activation functions, loss functions, and optimizers.
- `Model_def&train.py`: Contains the code for building, training, and evaluating the model.

## **Usage**
1. Clone the repository:
2. Explore `network.py` to understand the building blocks of the neural network.
3. Run `Model_def&train.py` to build and train the model for predicting sine function values.
4. Modify the code as needed to experiment with different configurations.

## **Conclusion**
This repository serves as an introduction to building neural networks from scratch without using a model class. It offers a hands-on experience with manual forward and backward passes, loss calculation, and optimization. For more details, refer to the regression section of the book "Neural Networks from Scratch."

Feel free to explore the code and share your feedback!
