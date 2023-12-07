import numpy as np
import cv2
import glob
from keras.models import load_model 
from keras.utils import to_categorical

# Global size variable
size = 300

# Load the pre-trained model
model = load_model('full_model.h5')

# Load Testing data: non-pothole
non_pothole_test_images = glob.glob("no_pot.jpg")
test2 = [cv2.imread(img, 0) for img in non_pothole_test_images]
for i in range(0, len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

# Load Testing data: potholes
pothole_test_images = glob.glob("pot.jpg")
test1 = [cv2.imread(img, 0) for img in pothole_test_images]
for i in range(0, len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

X_test = np.concatenate((temp3, temp4), axis=0)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)
y_test = np.concatenate((y_test1, y_test2), axis=0)
y_test = to_categorical(y_test)

print("")
X_test = X_test / 255.0
probabilities = model.predict(X_test)
predicted_classes = np.argmax(probabilities, axis=1)

print("")

for i, predicted_class in enumerate(predicted_classes):
    print(f">>> Predicted {i} = {predicted_class}")

print("")

metrics = model.evaluate(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(metrics[1] * 100))