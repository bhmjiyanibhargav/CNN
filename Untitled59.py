#!/usr/bin/env python
# coding: utf-8

# # question 01
Pooling in Convolutional Neural Networks (CNNs) serves two main purposes: 

**Purpose of Pooling**:

1. **Spatial Down-sampling**: Pooling reduces the spatial dimensions (width and height) of the feature maps. This helps in reducing the number of parameters and computations in the network, making it more computationally efficient.

2. **Feature Invariance**: Pooling helps create features that are invariant to small translations, rotations, and distortions. This is because the pooling operation looks for the most active features in a local neighborhood, which tend to be robust to small variations.

**Benefits of Pooling**:

1. **Reduced Computational Complexity**:
   - By down-sampling the feature maps, pooling reduces the number of computations in the network. This leads to faster training and inference times.

2. **Control Overfitting**:
   - Pooling helps in reducing overfitting by creating a more generalized representation of features. It introduces a degree of spatial invariance, which can improve the network's ability to generalize to unseen data.

3. **Increased Receptive Field**:
   - Pooling increases the receptive field of neurons in deeper layers. This allows them to capture more global information, which can be crucial for recognizing higher-level patterns.

4. **Improved Translation Invariance**:
   - Pooling helps in creating features that are robust to small translations. This is especially important in tasks like object recognition, where the exact location of features may vary.

5. **Memory Efficiency**:
   - Since pooling reduces the spatial dimensions of the feature maps, it also reduces the memory footprint of the network. This is important for training large models with limited GPU memory.

6. **Reduction of Overlapping Information**:
   - Pooling helps in capturing the most salient features while reducing redundant or overlapping information. This makes subsequent layers focus on more distinctive aspects of the data.

7. **Increased Depth**:
   - By reducing the spatial dimensions, pooling allows for more convolutional layers to be added to the network, which can potentially lead to a deeper and more expressive model.

8. **Preservation of Important Features**:
   - Pooling retains the most dominant features in the local regions, ensuring that important patterns are preserved.

It's important to note that while pooling has these benefits, there are also cases where it may not be necessary or may be replaced by other techniques like strided convolutions or global average pooling, depending on the specific requirements of the task and the architecture of the CNN.
# # question 02
Max pooling and min pooling are both types of pooling operations used in Convolutional Neural Networks (CNNs) to down-sample feature maps, reducing their spatial dimensions while retaining important information. However, they operate in slightly different ways:

**Max Pooling**:

1. **Operation**:
   - In max pooling, for each local region of the input feature map, the maximum value is taken. This maximum value becomes the representative value for that region.

2. **Advantages**:
   - Max pooling is effective in retaining the most dominant features in a region, which can be crucial for object recognition tasks.
   - It introduces a degree of translation invariance, as even if the position of a feature shifts slightly, the maximum value will still capture the presence of that feature.

3. **Downsides**:
   - It discards information about the non-maximal values in the region, potentially losing some nuanced information.
   - Max pooling can be sensitive to noise in the data.

**Min Pooling**:

1. **Operation**:
   - In min pooling, for each local region of the input feature map, the minimum value is taken. This minimum value becomes the representative value for that region.

2. **Advantages**:
   - Min pooling can be useful in scenarios where the smallest value in a region is more indicative of a certain feature or characteristic.

3. **Downsides**:
   - It discards information about the non-minimal values in the region, potentially losing some nuanced information.
   - Min pooling may be less common in practice compared to max pooling.

**Comparison**:

1. **Robustness to Noise**:
   - Max pooling tends to be less affected by noise because it focuses on the most dominant features in a region. Min pooling may be more sensitive to noise.

2. **Feature Detection**:
   - Max pooling is effective for detecting the presence of features, while min pooling may be more useful for detecting the absence or very low values of certain features.

3. **Application**:
   - Max pooling is widely used and considered a standard choice for pooling layers in CNN architectures.
   - Min pooling is less common and is usually chosen in specific scenarios where detecting minimum values is particularly relevant.

Ultimately, the choice between max pooling and min pooling (or other pooling methods) depends on the specific characteristics of the data and the requirements of the task at hand. In practice, max pooling is the more commonly used technique due to its effectiveness in capturing dominant features.
# # question 03
Padding in Convolutional Neural Networks (CNNs) is the process of adding extra, usually zero-valued, pixels around the boundaries of an image before applying a convolution operation. It is a technique used to control the spatial dimensions of the output feature maps.

Here are the key aspects of padding and its significance in CNNs:

**1. Preventing Information Loss**:

   - **Without Padding**: In a convolution operation without padding, the spatial dimensions of the output feature map are smaller than the input. This can lead to a reduction in spatial information as the convolution operation progresses through the layers.

   - **With Padding**: By adding extra pixels around the input image, padding ensures that the spatial dimensions of the output feature map remain the same or can be controlled. This helps in retaining more spatial information.

**2. Maintaining Spatial Dimensions**:

   - Padding allows you to preserve the spatial dimensions of the input feature map after convolution. This is particularly important when building deep architectures with multiple convolutional layers, as it helps in retaining finer spatial details.

**3. Addressing Boundary Effects**:

   - Without padding, the convolution operation can cause a loss of information at the edges of the input feature map. Padding helps mitigate this issue by ensuring that pixels at the boundaries are considered during the convolution.

**4. Facilitating Feature Extraction**:

   - Padding can be especially useful when the edges or boundaries of objects in the image are crucial for the task. It allows the CNN to extract features effectively from the entire image.

**5. Controlling Strides and Receptive Fields**:

   - Padding works in conjunction with the stride parameter. It allows you to control the effective stride, which can be important for determining the receptive field of neurons in deeper layers.

**6. Avoiding Vanishing/Exploding Gradients**:

   - In very deep networks, without padding, the spatial dimensions of the feature maps can become very small. This can lead to vanishing gradients during backpropagation. Padding helps in mitigating this issue.

**7. Types of Padding**:

   - There are two types of padding:
     - **Valid Padding (No Padding)**: No extra pixels are added. The output size is smaller than the input size.
     - **Same Padding (Zero Padding)**: Extra pixels are added around the input, making the output size the same as the input size.

**8. Computational Cost**:

   - While padding increases the amount of computation due to the increased size of the feature maps, it can be beneficial in terms of the model's ability to learn and represent spatial patterns effectively.

In summary, padding in CNNs is a critical technique for preserving spatial information, preventing boundary effects, and facilitating effective feature extraction. It plays a crucial role in the design and architecture of CNNs, allowing them to learn and represent complex spatial patterns in images.
# # question 04
It seems there may be some typos or misunderstandings in your question. It appears that you're asking about "CoXpace" and "contcast zeco-padding," which are not standard terms in the context of neural networks or machine learning.

However, I can provide information about "zero-padding" and "valid-padding" in the context of convolutional neural networks (CNNs) and their effects on the output feature map size:

**Zero-padding**:

1. **Definition**:
   - Zero-padding involves adding a border of zeros around the input image or feature map before applying the convolution operation.

2. **Effect on Output Size**:
   - Zero-padding allows the spatial dimensions of the output feature map to be the same as the input feature map. This is because the additional zeros around the input effectively expand its size.

3. **Benefits**:
   - Preserves spatial information, particularly important at the boundaries of objects.
   - Helps to prevent loss of information that can occur during convolution at the edges.

**Valid-padding**:

1. **Definition**:
   - Valid-padding (also known as no padding) involves applying the convolution operation without adding any extra border around the input.

2. **Effect on Output Size**:
   - Without padding, the spatial dimensions of the output feature map are reduced compared to the input.

3. **Impact on Feature Map Size**:
   - The reduction in size occurs because the convolution operation cannot be applied to the edges of the input without exceeding its boundaries.

In summary:

- **Zero-padding** maintains the spatial dimensions of the feature map, as it adds extra pixels around the input, effectively increasing its size.
- **Valid-padding** does not add any extra border, so the spatial dimensions of the output feature map are smaller than the input.

These padding techniques are crucial in controlling the spatial dimensions of feature maps as they pass through convolutional layers in a CNN. They help preserve spatial information, control receptive fields, and prevent loss of information at the boundaries.
# # TOPIC: Exploring LeNet

# # question 01

# # LeNet-5 is a convolutional neural network (CNN) architecture developed by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. It was introduced in 1998 and is considered one of the pioneering CNN architectures. LeNet-5 was designed for handwritten digit recognition tasks, particularly for reading zip codes, digits on bank checks, and similar applications.
# 
# Here is an overview of the LeNet-5 architecture:
# 
# **1. Input Layer**:
#    - The LeNet-5 architecture takes grayscale images as input, with each image having a size of 32x32 pixels.
# 
# **2. First Convolutional Layer**:
#    - Convolutional Layer 1:
#      - Filter Size: 5x5
#      - Number of Filters: 6
#      - Activation Function: Tanh
#      - Padding: Valid
#      - Stride: 1
#    - This layer applies six convolutional filters to the input images, resulting in six feature maps.
# 
# **3. First Pooling Layer**:
#    - Max Pooling Layer 1:
#      - Pooling Size: 2x2
#      - Stride: 2
#    - This layer performs max pooling, reducing the spatial dimensions of the feature maps and enhancing translation invariance.
# 
# **4. Second Convolutional Layer**:
#    - Convolutional Layer 2:
#      - Filter Size: 5x5
#      - Number of Filters: 16
#      - Activation Function: Tanh
#      - Padding: Valid
#      - Stride: 1
#    - This layer applies sixteen convolutional filters to the feature maps from the first pooling layer, resulting in sixteen feature maps.
# 
# **5. Second Pooling Layer**:
#    - Max Pooling Layer 2:
#      - Pooling Size: 2x2
#      - Stride: 2
#    - Similar to the first pooling layer, this layer performs max pooling, further reducing the spatial dimensions.
# 
# **6. Fully Connected Layers**:
#    - Flatten Layer:
#      - This layer flattens the output from the previous layer into a one-dimensional vector.
#    - Fully Connected Layer 1:
#      - Number of Neurons: 120
#      - Activation Function: Tanh
#    - Fully Connected Layer 2:
#      - Number of Neurons: 84
#      - Activation Function: Tanh
#    - Fully Connected Output Layer:
#      - Number of Neurons: 10 (corresponding to the 10 possible digits 0-9)
#      - Activation Function: Softmax
#    - The fully connected layers process the high-level features extracted by the convolutional layers and perform classification.
# 
# **7. Output Layer**:
#    - The output layer uses a softmax activation function to produce class probabilities for the ten possible digits.
# 
# **Summary**:
# 
# - LeNet-5 consists of two convolutional layers followed by max pooling layers, which help extract hierarchical features from the input images.
# - Fully connected layers at the end process the features for classification.
# - The use of the tanh activation function was common in earlier neural networks.
# - The architecture is designed for efficient training and inference on handwritten digit recognition tasks.
# 
# LeNet-5 laid the foundation for modern CNN architectures and played a significant role in popularizing the use of convolutional networks for computer vision tasks.

# # question 02
LeNet-5, designed by Yann LeCun and his team, is a pioneering Convolutional Neural Network (CNN) architecture that was introduced in 1998. It was primarily developed for handwritten digit recognition, particularly in the context of reading zip codes and digits on checks. Here are the key components of LeNet-5 and their respective purposes:

1. **Input Layer**:
   - **Purpose**: This is the initial layer that takes the input image. In the case of LeNet-5, the input images are grayscale and have a size of 32x32 pixels.

2. **Convolutional Layers**:
   - **Purpose**: These layers perform the convolution operation, which involves sliding a small filter (also called a kernel) across the input image to extract features. Each convolutional layer has specific characteristics:
     - **Filter Size**: Specifies the dimensions of the filter (e.g., 5x5).
     - **Number of Filters**: Determines how many different features are detected in each layer.
     - **Activation Function (Tanh)**: The hyperbolic tangent activation function was used in LeNet-5. It introduces non-linearity into the network, allowing it to learn complex relationships.

3. **Pooling Layers**:
   - **Purpose**: Pooling layers down-sample the spatial dimensions of the feature maps produced by the convolutional layers. This reduces computational complexity and helps in creating a hierarchy of features. LeNet-5 uses average pooling.

4. **Fully Connected Layers**:
   - **Purpose**: These layers are densely connected to all the neurons in the previous layer. They process the high-level features extracted by the convolutional layers and perform classification. In LeNet-5:
     - The first fully connected layer has 120 neurons with a hyperbolic tangent activation function.
     - The second fully connected layer has 84 neurons with a hyperbolic tangent activation function.
     - The output layer has 10 neurons (one for each possible digit) with a softmax activation function for multi-class classification.

5. **Flatten Layer**:
   - **Purpose**: This layer is used to convert the multi-dimensional output from the last pooling layer into a one-dimensional vector. This prepares the data for the fully connected layers.

6. **Output Layer**:
   - **Purpose**: The output layer produces the final prediction for the input image. It typically uses a softmax activation function for multi-class classification tasks, which gives probabilities for each class.

**Summary**:

- LeNet-5 is characterized by a sequence of convolutional layers followed by pooling layers, which helps in feature extraction and dimensionality reduction.
- Fully connected layers at the end process the features for classification.
- The use of the hyperbolic tangent activation function was common in earlier neural networks.

LeNet-5 played a crucial role in demonstrating the effectiveness of convolutional neural networks for computer vision tasks and has influenced the development of numerous subsequent architectures.
# # question 03
**Advantages of LeNet-5:**

1. **Effective Feature Extraction**:
   - LeNet-5 was one of the first CNN architectures to demonstrate the effectiveness of hierarchical feature extraction through convolutional layers. It showed that deep learning models can automatically learn meaningful features from raw pixel data.

2. **Translation Invariance**:
   - The combination of convolutional layers and pooling layers in LeNet-5 allows it to be robust to small translations in the input image. This is a crucial property for image recognition tasks.

3. **Sparse Connectivity**:
   - LeNet-5 introduces sparse connectivity patterns by using a small receptive field and shared weights. This significantly reduces the number of parameters compared to fully connected networks, making it computationally efficient.

4. **Efficient Architecture**:
   - LeNet-5 was designed to be efficient in terms of memory and computation. It was able to achieve high performance even on the computing resources available at the time of its development.

5. **Architectural Pioneering**:
   - LeNet-5 played a pivotal role in popularizing CNNs for computer vision tasks. It laid the foundation for subsequent advancements in CNN architectures and their widespread adoption in the field of deep learning.

**Limitations of LeNet-5:**

1. **Limited Complexity**:
   - Compared to modern deep learning architectures, LeNet-5 is relatively shallow. It may struggle with more complex tasks or datasets that require deeper networks to capture intricate features.

2. **Activation Function (Tanh)**:
   - LeNet-5 uses the hyperbolic tangent activation function (tanh), which saturates for extreme values, potentially leading to the vanishing gradient problem. Modern architectures often use activation functions like ReLU, which mitigate this issue.

3. **Limited to Small Images**:
   - LeNet-5 was designed for small 32x32 pixel images. While it worked well for its intended applications (digit recognition), it may not be suitable for tasks that require processing larger images.

4. **Limited to Single Channel Images**:
   - LeNet-5 was designed for grayscale images. Adapting it to handle multi-channel (e.g., RGB) images would require modifications to the architecture.

5. **Subsequent Architectural Advances**:
   - Since the introduction of LeNet-5, there have been numerous architectural advancements, such as deeper networks, skip connections, and attention mechanisms, which have led to even more powerful models for image classification tasks.

6. **Training on Modern Hardware**:
   - LeNet-5 was designed for the computing resources available in the late 1990s. Training it on modern hardware with powerful GPUs may not fully leverage the capabilities of contemporary deep learning frameworks.

In summary, LeNet-5 was a groundbreaking architecture that significantly contributed to the advancement of computer vision tasks, especially in the context of handwritten digit recognition. However, due to its architectural simplicity, it may not be the most suitable choice for complex tasks or large-scale image datasets. Modern architectures have since addressed many of these limitations, leading to state-of-the-art performance in a wide range of computer vision applications.
# # question 04

# In[3]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension to the images (since LeNet-5 expects input shape of (32, 32, 1))
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Define the LeNet-5 model
model = models.Sequential()

# First Convolutional Layer
model.add(layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1), padding='valid'))
model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))

# Second Convolutional Layer
model.add(layers.Conv2D(16, (5, 5), activation='tanh', padding='valid'))
model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Layers
model.add(layers.Dense(120, activation='tanh'))
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_accuracy*100:.2f}%")



# # TOPIC: Analyzing AlexNet

# # question 01
AlexNet is a deep convolutional neural network (CNN) architecture designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, marking a significant breakthrough in computer vision tasks. Here's an overview of the AlexNet architecture:

**1. Input Layer**:
   - The network takes an input image of size 224x224 pixels with three color channels (RGB).

**2. Convolutional Layers**:
   - **Convolutional Layer 1**:
     - Filter Size: 11x11
     - Number of Filters: 96
     - Activation Function: ReLU
     - Stride: 4
     - Padding: Valid
   - **Convolutional Layer 2**:
     - Filter Size: 5x5
     - Number of Filters: 256
     - Activation Function: ReLU
     - Stride: 1
     - Padding: Same
   - **Convolutional Layer 3**:
     - Filter Size: 3x3
     - Number of Filters: 384
     - Activation Function: ReLU
     - Stride: 1
     - Padding: Same
   - **Convolutional Layer 4**:
     - Filter Size: 3x3
     - Number of Filters: 384
     - Activation Function: ReLU
     - Stride: 1
     - Padding: Same
   - **Convolutional Layer 5**:
     - Filter Size: 3x3
     - Number of Filters: 256
     - Activation Function: ReLU
     - Stride: 1
     - Padding: Same
   - **Purpose**: These convolutional layers are responsible for feature extraction. They apply a series of convolution operations to learn hierarchical features.

**3. Max Pooling Layers**:
   - After each of the first two convolutional layers, there is a max pooling layer with a pooling size of 3x3 and a stride of 2. These layers help reduce spatial dimensions and enhance translation invariance.

**4. Fully Connected Layers**:
   - **Fully Connected Layer 1**:
     - Number of Neurons: 4096
     - Activation Function: ReLU
   - **Fully Connected Layer 2**:
     - Number of Neurons: 4096
     - Activation Function: ReLU
   - **Fully Connected Layer 3 (Output Layer)**:
     - Number of Neurons: 1000 (corresponding to 1000 classes in the ImageNet dataset)
     - Activation Function: Softmax
   - **Purpose**: These fully connected layers process the high-level features and perform classification.

**5. Dropout Layers**:
   - After the first two fully connected layers, there are dropout layers with a dropout rate of 0.5. These layers help reduce overfitting by randomly setting a fraction of input units to zero during training.

**6. Output Layer**:
   - The output layer uses a softmax activation function to produce class probabilities for the 1000 possible classes in the ImageNet dataset.

**Summary**:
- AlexNet is characterized by its deep architecture, which includes multiple convolutional layers, max pooling layers, and fully connected layers.
- The use of the rectified linear unit (ReLU) activation function helps address the vanishing gradient problem.
- Dropout is employed to reduce overfitting, making the model more robust.

AlexNet significantly advanced the state of the art in computer vision and paved the way for even deeper and more complex CNN architectures. It demonstrated the power of deep learning for image classification tasks and played a crucial role in the resurgence of neural networks. 
# # question 02
AlexNet introduced several architectural innovations that contributed to its breakthrough performance in computer vision tasks. These innovations addressed key challenges in training deep neural networks and significantly improved their ability to learn and represent complex features. Here are the key innovations of AlexNet:

**1. Deep Architecture**:
   - AlexNet was one of the first deep convolutional neural networks (CNNs) with multiple layers. It had eight layers, which was considered very deep at the time (2012). This depth allowed the network to learn hierarchical features at different levels of abstraction.

**2. Rectified Linear Unit (ReLU) Activation Function**:
   - AlexNet used the ReLU activation function instead of traditional activation functions like sigmoid or tanh. ReLU activations help mitigate the vanishing gradient problem, allowing for more efficient training of deep networks. They also accelerate convergence during gradient descent.

**3. Local Response Normalization (LRN)**:
   - AlexNet introduced a form of local response normalization after the ReLU activation in the convolutional layers. LRN helps neurons respond more strongly to specific patterns, improving generalization. However, later research found that LRN is not always necessary and Batch Normalization became more popular.

**4. Overlapping Pooling Layers**:
   - The pooling layers in AlexNet used a relatively large pooling window (3x3) with a stride of 2. This overlapping pooling strategy helped in reducing spatial dimensions while retaining more information. It contributed to better translation invariance.

**5. Data Augmentation**:
   - AlexNet employed extensive data augmentation techniques during training. This involved applying random transformations to the training images (such as cropping, flipping, and color alterations) to create variations of the dataset. This helped the model generalize better to unseen data.

**6. Dropout Regularization**:
   - AlexNet incorporated dropout layers after the first two fully connected layers. Dropout randomly sets a fraction of the input units to zero during training. This helps prevent overfitting by reducing co-adaptation of neurons.

**7. Large-Scale Training**:
   - Training AlexNet required significant computational resources, and it was one of the first models to be trained on powerful GPUs. This allowed for faster training times, making it feasible to train deep networks on large-scale datasets.

**8. GPU Acceleration**:
   - AlexNet was one of the first models to demonstrate the potential of GPU acceleration for training deep neural networks. This greatly sped up the training process compared to traditional CPU-based training.

These architectural innovations collectively contributed to the breakthrough performance of AlexNet, allowing it to significantly outperform previous state-of-the-art models in image classification tasks. The success of AlexNet demonstrated the potential of deep learning and led to a renaissance in neural network research and applications in computer vision.
# # question 03
In the architecture of AlexNet, each type of layer (convolutional, pooling, and fully connected) plays a distinct and crucial role in feature extraction, dimensionality reduction, and classification. Here's a detailed discussion of the role of each type of layer in AlexNet:

**1. Convolutional Layers**:

   - **Role**:
     - Convolutional layers are responsible for feature extraction. They apply a series of convolution operations to the input image or feature map, using learnable filters. These filters detect various features in the input data, such as edges, textures, and more complex patterns.

   - **In AlexNet**:
     - AlexNet uses five convolutional layers. The first layer applies large 11x11 filters to capture low-level features, while subsequent layers apply smaller 3x3 and 5x5 filters to capture more complex and abstract features.

   - **Non-Linearity (ReLU)**:
     - After each convolutional operation, the ReLU activation function is applied. ReLU introduces non-linearity, allowing the network to model complex relationships in the data.

   - **Response Normalization (LRN)** (Note: Not widely used today):
     - After the ReLU activation in the first two convolutional layers, AlexNet applies local response normalization (LRN). This operation enhances the response of neurons to specific patterns.

**2. Pooling Layers**:

   - **Role**:
     - Pooling layers down-sample the spatial dimensions of the feature maps produced by the convolutional layers. This reduces the computational complexity of the network and enhances translation invariance.

   - **In AlexNet**:
     - AlexNet employs max pooling after the first and second convolutional layers. The pooling size is 3x3 with a stride of 2, leading to a reduction in spatial dimensions.

   - **Overlapping Pooling**:
     - One distinctive feature of AlexNet's pooling layers is that they use overlapping pooling. This means that the pooling windows overlap, allowing the network to retain more spatial information.

**3. Fully Connected Layers**:

   - **Role**:
     - Fully connected layers process the high-level features extracted by the convolutional layers and perform classification.

   - **In AlexNet**:
     - AlexNet has three fully connected layers. The first two have 4096 neurons with ReLU activation, and the third serves as the output layer with 1000 neurons (corresponding to the classes in the ImageNet dataset) and a softmax activation function.

   - **Dropout**:
     - After the first two fully connected layers, dropout with a rate of 0.5 is applied. Dropout helps reduce overfitting by randomly deactivating neurons during training.

   - **Output Layer**:
     - The output layer produces class probabilities for the 1000 possible classes in the ImageNet dataset.

**Summary**:

- Convolutional layers extract features from the input data.
- Pooling layers reduce spatial dimensions and enhance invariance to translation.
- Fully connected layers process high-level features for classification.

These layers work together to create a deep and effective architecture for image classification tasks, as demonstrated by the success of AlexNet on the ImageNet dataset.
# # question 04

# In[ ]:


import tensorflow as tf
from tensorflow.keras import models, layers

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the AlexNet model
model = models.Sequential()

# Convolutional Layers
model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3), padding='valid'))
model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Layers
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_accuracy*100:.2f}%")


# In[ ]:




