import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from sklearn.utils import shuffle
import os

# Load the pre-trained model
model = load_model('full_model.h5')  # Update with the correct path to your model file

size = 300

# Load Testing data: Plain
plain_test_path = "My Dataset/test/Plain"
plain_test_images = [os.path.join(plain_test_path, img) for img in os.listdir(plain_test_path) if img.endswith(".jpg")]
test_plain = [cv2.imread(img, 0) for img in plain_test_images]
for i in range(len(test_plain)):
    test_plain[i] = cv2.resize(test_plain[i], (size, size))
temp_plain = np.asarray(test_plain)

# Load Testing data: Pothole
pothole_test_path = "My Dataset/test/Pothole"
pothole_test_images = [os.path.join(pothole_test_path, img) for img in os.listdir(pothole_test_path) if img.endswith(".jpg")]
test_pothole = [cv2.imread(img, 0) for img in pothole_test_images]
for i in range(len(test_pothole)):
    test_pothole[i] = cv2.resize(test_pothole[i], (size, size))
temp_pothole = np.asarray(test_pothole)

X_test = []
X_test.extend(temp_plain)
X_test.extend(temp_pothole)
X_test = np.asarray(X_test)

X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_test_plain = np.zeros([temp_plain.shape[0]], dtype=int)
y_test_pothole = np.ones([temp_pothole.shape[0]], dtype=int)

y_test = []
y_test.extend(y_test_plain)
y_test.extend(y_test_pothole)
y_test = np.asarray(y_test)

y_test = to_categorical(y_test)

# Shuffle the data
X_test, y_test = shuffle(X_test, y_test)

# Normalize the pixel values
X_test = X_test / 255.0

# Evaluate the model on the test data
metrics = model.evaluate(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(metrics[1] * 100))

# Make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Display classification report and confusion matrix
print("\nClassification Report:\n", classification_report(np.argmax(y_test, axis=1), predicted_labels))
print("\nConfusion Matrix:\n", confusion_matrix(np.argmax(y_test, axis=1), predicted_labels))
