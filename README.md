# Hand Gesture Recognition

## Data Collection and Storage

* Collected hand gesture data is saved to a CSV file named `landmarks_medium.csv` in the same directory as the script.
* Each row in the CSV file represents a single hand gesture, with columns representing the x and y coordinates of the hand landmarks.
* The CSV file is used to train the logistic regression model.

## Model Training and Storage

* The trained logistic regression model is saved to a file named `gesture_model.sav` using Pickle.
* The model is trained using the data from `landmarks_medium.csv` and can recognize 30 different hand gestures (A-Z, HELLO, NICE, BAD, and GOOD).

## Real-time Hand Gesture Recognition

* The `(implement.py)` script loads the trained model from `gesture_model.sav` and uses it for real-time hand gesture recognition.
* The script captures hand gestures using OpenCV and MediaPipe, extracts landmarks, and uses the trained model to predict the gesture.
* The recognized gesture is displayed on the screen in real-time.

## File Structure

* `(collection.py)`: Script for collecting hand gesture data.
* `(train.py)`: Script for training the logistic regression model.
* `(implement.py)`: Script for real-time hand gesture recognition.
* `landmarks_medium.csv`: CSV file containing collected hand gesture data.
* `gesture_model.sav`: File containing the trained logistic regression model.

## Dependencies

* OpenCV
* MediaPipe
* scikit-learn
* Pickle

## Usage

1. Run `(collection.py)` to collect hand gesture data.
2. Run `(train.py)` to train the logistic regression model.
3. Run `(implement.py)` for real-time hand gesture recognition.

