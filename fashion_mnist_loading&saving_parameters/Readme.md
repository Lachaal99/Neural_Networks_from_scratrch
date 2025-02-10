# Neural Network Classification Model with Fashion MNIST (Hyperparameter Management)

This repository contains a classification model for the Fashion MNIST dataset. It builds on the existing Fashion MNIST example in the repository, with the addition of methods to save and load the model's hyperparameters( exemple guided by the book). This new functionality makes it easier to reuse and adjust model settings without retraining from scratch.

## **Project Overview**
- **Dataset** : Fashion MNIST — 70,000 grayscale images (28x28) of 10 clothing categories such as shirts, sneakers, and dresses.
- **New Feature**: Added methods to save and load hyperparameters, enabling efficient model management.
- **Pre-Saved Hyperparameters**: A pre-saved set of hyperparameters is available for immediate use.
## **Files in This Repository**
- network_fully.py: Contains the neural network building blocks and model class (dense layers, activation functions, loss functions, optimizers).
- model.py : Dataloading, model buildup , training and saving hyperparams .  
- loading_params.py: Implements loading of model hyperparameters.
- fashion_mnist.params: Contains a pre-saved set of hyperparameters.
## **Usage Instructions**
- **Clone the repository**
- Download the Fashion MNIST dataset if you want to run the full training process.
- Use loading_params.py to load or save the model’s hyperparameters.
- Modify and test the model using the pre-saved hyperparameters or your own configurations.
## **Conclusion**
This project builds on the Fashion MNIST classification model by adding robust hyperparameter management. You can easily load, save, and reuse hyperparameters, making your workflow more efficient and flexible.

Feel free to explore, modify, and contribute! Feedback is always welcome. 