# CNN Model for Pizza vs. Not Pizza Classification

This project demonstrates a basic convolutional neural network (CNN) model for binary classification between images of pizza and non-pizza objects. The dataset used contains two classes, and the model is trained and evaluated to predict the class of the images. 

## Key Concepts:
- **Overfitting**: When the model performs well on the training data but poorly on unseen data (test data). This happens when the model learns patterns specific to the training data, including noise, but fails to generalize to new, unseen data.
- **Underfitting**: When the model is too simple and does not capture the underlying patterns in the training data, resulting in poor performance on both the training and test data.

In this project, we observe **overfitting** because the model's performance on the training dataset is significantly better than on the test dataset.

## Dataset
The dataset contains images divided into two categories:
- **Train Set**: 1474 images
- **Test Set**: 492 images

The images are rescaled using `ImageDataGenerator` with a normalization factor of 1/255 to scale pixel values between 0 and 1. This helps improve the model's performance by making the training more stable.

## CNN Architecture

The architecture of the model is as follows:
1. **Input Layer**: The input images have a shape of (224, 224, 3) (height, width, and 3 color channels).
2. **Convolutional Layers**: Three `Conv2D` layers are applied to extract features from the images, each followed by a **MaxPooling** layer to downsample the image and reduce dimensionality.
3. **Flattening Layer**: Converts the 2D feature maps into a 1D vector to feed into the fully connected layers.
4. **Dense Layers**:
   - First fully connected layer with 100 neurons and ReLU activation function.
   - Output layer with 1 neuron and a sigmoid activation function for binary classification.
   
## Training and Results

The model is compiled using:
- **Optimizer**: Adam (a popular optimization algorithm)
- **Loss Function**: Binary Crossentropy (suitable for binary classification tasks)
- **Metrics**: Accuracy

### Model Training
The model is trained for 7 epochs on the training dataset:

```python
model_saved = model.fit_generator(traindataset, epochs=7)
```

### Model Evaluation
The model is evaluated on both the training and test datasets:

1. **Test Set Results**:
    ```python
    Test loss: 0.5352
    Test accuracy: 75.6%
    ```

2. **Training Set Results**:
    ```python
    Train loss: 0.1777
    Train accuracy: 94.2%
    ```

### Observing Overfitting
From the evaluation results, we can observe that:
- The **training accuracy** is high (94.2%) with a **low training loss** (0.1777), which means the model fits well to the training data.
- However, the **test accuracy** is significantly lower (75.6%) with a **higher test loss** (0.5352), indicating that the model is overfitting to the training data and not generalizing well to the test data.

### Overfitting Explanation
- The model performs well on the training data, meaning it has learned patterns specific to that data.
- The high test loss and lower test accuracy show that the model struggles to generalize to unseen data, a sign of overfitting.
- **Possible Solution**: To mitigate overfitting, we can try:
  - Adding **regularization** like dropout layers to randomly turn off some neurons during training.
  - **Data Augmentation** to create variations in the dataset (rotation, scaling, flipping) so that the model learns to generalize better.
  - **Early stopping** to halt the training process when the performance on validation data begins to deteriorate.

## Data Augmentation
To address overfitting, we can introduce more data variations using data augmentation techniques like rotation, zooming, horizontal flipping, etc., which will generate new samples based on the existing ones. This can help the model generalize better to unseen data.

```python
train=ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

## Conclusion
This project demonstrates a basic CNN model that shows signs of **overfitting** due to a higher performance on the training data compared to the test data. To achieve better generalization, adding more data variation through augmentation, dropout, or other regularization techniques can be effective in preventing the model from memorizing the training data.

 