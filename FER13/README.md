# Classification Model for FER2013 Dataset Using Only Dense Layers

In this project, I tried applying what I learned from the "Neural Networks from Scratch" book on a more complex, real-world dataset: the FER2013 dataset for facial emotion recognition. While I know that using only dense layers to solve a computer vision problem is not the most efficient approach due to the high dimensionality of image data and spatial dependencies, my goal was to experience the difference between Convolutional Neural Networks (CNNs) and fully connected networks (dense layers). I aimed to understand the fundamental differences between the convolutional layer and dense layers, and how they perform in different situations.

### Key Points:
- **Why dense layers?**  
  Although it’s not ideal for image classification (since dense layers rely on predictions based on individual pixels), the aim was to explore how a fully connected network performs in contrast to CNNs.
  
- **Key Features Implemented:**  
  - Added a **validation step** during training to evaluate the model on validation data at each epoch (labeled as "test data").
  - Implemented a **history tracking method** similar to those in advanced frameworks to track loss and accuracy over different epochs for both training and validation data.

### Files Included:
- **model_notebook.ipynb:** Where the preprocessing and training took place.
- **Network_full.py:** The raw model code where all the building blocks are implemented.
- **fer13.model:** A saved copy of the trained model.

### Results:
As expected, the fully dense model didn't perform well due to the high dimensionality of the data, spatial dependencies, and varying features within the images. This experiment highlighted the need for specialized methods and layers, such as Convolutional Layers, which are designed to better handle image data by focusing on feature dimensionality and spatial relationships.

### Next Steps:
In the next project, we will implement a **Convolutional Neural Network (CNN)** to improve performance in image data classification, taking advantage of the specialized layers to handle the complexities of image data.

---

Feel free to explore the repository and try running the code to further understand how the dense layers behave when applied to image classification problems!(don't forget to download the dataset)

