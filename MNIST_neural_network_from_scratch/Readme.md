# Handwritten Digit Recognition Model

This repository contains a neural network model for predicting handwritten digits. The model is built exclusively with fully connected dense layers and includes the full process, from data preprocessing to training and inference. Below is a detailed description of the contents and features:

## **Project Overview**
- **Model Structure**: The model is implemented using fully connected dense layers.
- **Data Preprocessing**: Data is preprocessed and prepared for training within the `model.ipynb` notebook.
- **Fundamental Building Blocks**: All core components of the network are included in `network_full.py`, which contains the building blocks I developed for neural networks using the "Neural Networks from Scratch" book. The `model.ipynb` file contains the implementation of the neural network with the following structure:
  - A dense layer with 256 nodes using the ReLU activation function.
  - A dropout layer to reduce the risk of overfitting.
  - A dense layer with 128 nodes (ReLU activation function).
  - Another dropout layer.
  - A dense layer with 10 nodes and a Softmax activation function for classification and normalization.

  For loss calculation, we used the **Categorical Cross Entropy** method, and the **Adam Optimizer** was chosen for optimization. You can find the optimizer's parameters inside the code.

## **Key Features**
- **Model Saving and Loading**: The model class includes methods for saving and loading trained models. This allows you to reuse the model at any time without retraining.
- **Inference Examples**: After loading the saved model, I generated two sample images to demonstrate the inference process(test1 & test2).

## **Main Files in This Repository**
- `model.ipynb`: The main notebook that contains the full process from preprocessing to training and inference.
- `network_full.py`: Contains the core implementation of the neural network.
-  Folders containing the mnist datasets.

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/M.git
   ```
2. Run the `model.ipynb` notebook to preprocess data, train the model, and perform inference (feel free to customize it yourself).
3. Use the model saving and loading functions as needed to save your trained model and load it for future use (there's already a pre-saved model).

## **Conclusion**
This repository serves as a hands-on project for predicting handwritten digits using a neural network with fully connected layers. The added functionality for saving and loading models makes it easier to manage and reuse your trained models.

Feel free to explore the code and contribute! Feedback and suggestions are always welcome.
