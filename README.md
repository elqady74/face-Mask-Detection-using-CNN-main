# face-Mask-Detection-using-CNN
## 📌 Project Overview
This project focuses on **Face Mask Detection** using **Convolutional Neural Networks (CNNs)**. The model is trained to classify images of people **wearing a mask** and **without a mask** using deep learning techniques.

## 🚀 Features
- Uses **CNN architecture** for image classification.
- Data preprocessing and augmentation to improve model accuracy.
- Implements **TensorFlow** and **Keras** for training and evaluation.
- Real-time prediction capability using OpenCV.

## 📂 Dataset
The dataset is sourced from **Kaggle**: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

It consists of:
- **3,725** images of people wearing masks
- **3,828** images of people without masks

## 🔧 Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/YoussefAbdelnasser11/face-Mask-Detection-using-CNN.git
cd face-Mask-Detection-using-CNN
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset from Kaggle
```python
!kaggle datasets download -d omkargurav/face-mask-dataset
```
Extract the dataset and place it inside the project directory.

## 🏗 Model Architecture
The CNN model consists of:
- **Convolutional Layers** for feature extraction
- **MaxPooling Layers** to reduce spatial dimensions
- **Flatten & Dense Layers** for classification
- **Dropout Layers** to prevent overfitting

```python
num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))
```

## 🏋️ Training the Model
```python
history = model.fit(X_train_scaled, Y_train, validation_split=0.15, epochs=5)
```

## 🧐 Model Evaluation
```python
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy=', accuracy)
```

## 🎯 Real-Time Prediction
You can test the model with an image input:
```python
input_image = cv2.imread("test_image.jpg")
input_image_resized = cv2.resize(input_image, (128,128))
input_image_scaled = input_image_resized/255
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
print("Prediction:", np.argmax(input_prediction))
```

## 📊 Results
- **Training Accuracy:** ~94%
- **Test Accuracy:** ~91%
- **Challenges:** Overfitting, class imbalance, tuning hyperparameters

## 📜 Future Improvements
- Use **Transfer Learning** (e.g., EfficientNet, MobileNet) for better accuracy.
- Improve dataset balance and augmentation.
- Deploy the model as a **web or mobile app**.

## 🤝 Contributing
Feel free to contribute by opening an issue or submitting a pull request! 🚀

## 📄 License
This project is licensed under the **MIT License**.

---
### 🔗 Connect with Me
- **LinkedIn:** [Yousef Abdalnasser](https://www.linkedin.com/in/yousefabdalnasser)

🔹 *If you find this project useful, don't forget to ⭐ the repo!*
