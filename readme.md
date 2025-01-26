# Neural Networks from Scratch in Python: My Learning Journey

This repository documents my journey of learning and implementing neural networks by following the book *"Neural Networks from Scratch in Python"*. Using only NumPy, I developed the core components required to build and train neural networks from scratch. Below is a detailed breakdown of what I implemented:

## **Core Components**
### 1. **Layers**
- Input Layer, Hidden Layers, and Output Layer with fully connected weights and biases.

### 2. **Activation Functions**
- **ReLU** (Rectified Linear Unit): Introduces non-linearity and handles vanishing gradients.
- **Softmax**: Used for multi-class classification, providing probabilities.
- **Sigmoid**: Common for binary classification tasks.
- **Linear Activation**: Typically used for regression problems.

### 3. **Loss Functions**
- **Categorical Cross-Entropy**: For multi-class classification problems.
- **Binary Cross-Entropy**: For binary logistic regression.
- **Mean Squared Error (MSE)**: For regression tasks.
- **Mean Absolute Error (MAE)**: Another metric for regression tasks.

### 4. **Optimizers**
- **Stochastic Gradient Descent (SGD)**:
  - Includes learning rate decay (adjusting the learning rate over epochs).
  - Momentum (takes the direction of previous updates into account).
- **Adaptive Gradient Descent (Adagrad)**:
  - Adjusts updates based on the magnitude of gradients to prevent drastic parameter changes.
- **Root Mean Squared Propagation (RMSprop)**:
  - Similar to Adagrad but introduces a decay factor for smoothing updates.
- **Adaptive Moment Estimation (Adam)**:
  - Combines features of learning rate decay, momentum, and adaptive updates for efficient optimization. This optimizer is widely used due to its versatility.

## **Regularization Techniques**
To prevent overfitting and improve generalization:
1. **L1 and L2 Regularization**:
   - Adds a penalty to large weights during training to avoid overfitting.
2. **Dropout Layer**:
   - Randomly nullifies some neurons' outputs during training to ensure the network does not overly rely on specific neurons, promoting generalization.

## **Final Implementation**
After building all the individual components, I integrated them into a **Model Class**. This class automates forward passes, backpropagation, training, and validation, mimicking the functionality of popular frameworks like TensorFlow or PyTorch—all implemented using only NumPy.

## **Examples**
The examples in this repository are directly inspired by the book. Each example demonstrates a neural network solving a specific problem. Datasets and detailed implementations are also included, with code well-documented for clarity and reference.

---

### **Feedback**
- If you're reviewing this project, I'd appreciate feedback on:
  - Code optimization and readability.
  - Suggestions for additional features or improvements.
  - Any corrections to the implementations.

---

### **Conclusion**
This repository reflects my hands-on learning experience with neural networks. By building everything from scratch, I deepened my understanding of the inner workings of neural networks and how different components interact during training.
