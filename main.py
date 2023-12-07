import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os  

size = 300
inputShape = (size, size, 1)
num_classes = 2


def kerasModel4():
    model = tf.keras.Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

base_dir = "C:/Users/avina/OneDrive/Desktop/pothole-detection-system-using-convolution-neural-networks-master/pothole-detection-system-using-convolution-neural-networks-master/Real-time Files/My Dataset/train"
class_names = os.listdir(base_dir)

train_images = []
train_labels = []

for class_name in class_names:
    class_dir = os.path.join(base_dir, class_name)
    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300))  # Resize images as needed
        train_images.append(img)
        train_labels.append(class_names.index(class_name))

# Convert to numpy arrays
X = np.array(train_images)
y = np.array(train_labels)

# Shuffle and split the dataset
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

print("Train shape X:", X_train.shape)
print("Train shape y:", y_train.shape)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the CNN model
model = kerasModel4()

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


print("")

# Evaluate on training set
metrics_train = model.evaluate(X_train, y_train)
print("Training Accuracy: {:.2f}%".format(metrics_train[1] * 100))

print("")

# Evaluate on test set
metrics_test = model.evaluate(X_test, y_test)
print("Testing Accuracy: {:.2f}%".format(metrics_test[1] * 100))

print("Saving model weights and configuration file")
model.save('latest_full_model.h5')
print("Saved model to disk")
