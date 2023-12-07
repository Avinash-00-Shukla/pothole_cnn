import cv2
import imutils 
from keras.models import load_model

# Set the global size variable
size = 300

# Load the pre-trained model
loadedModel = load_model('full_model.h5')

# Resize the frame and make predictions
def predict_pothole(current_frame):
    current_frame = cv2.resize(current_frame, (size, size))
    current_frame = current_frame.reshape(1, size, size, 1).astype('float') / 255.0
    prob = loadedModel.predict(current_frame)
    max_prob = max(prob[0])
    
    if max_prob > 0.90:
        return loadedModel.predict(current_frame)[0], max_prob
    return "none", 0

# Main function
if __name__ == '__main__':
    # Open the camera
    camera = cv2.VideoCapture(0)

    show_pred = False

    # Loop until interrupted
    while True:
        grabbed, frame = camera.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        # Make a clone of the frame
        clone = frame.copy()

        # Convert the frame to grayscale
        gray_clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

        # Predict pothole and get probability
        pothole, prob = predict_pothole(gray_clone)

        keypress_toshow = cv2.waitKey(1)

        if keypress_toshow == ord("e"):
            show_pred = not show_pred

        if show_pred:
            text = f"{pothole} {prob*100:.2f}%"
            cv2.putText(clone, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("GrayClone", gray_clone)
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()
