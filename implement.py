import cv2
import mediapipe as mp
import time
import pickle

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

sTime = time.time()

pivot_x = 0
pivot_y = 0

row_list = []

font = cv2.FONT_HERSHEY_SIMPLEX
org = (185, 50)
color = (0, 0, 255)

model_file_name = 'gesture_model.sav'
model = pickle.load(open(model_file_name, 'rb'))

gesture_labels = {
    0: 'A',
    1: 'B',
    2: 'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'J',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y',
    25:'Z',
    26:'HELLO',
    27:'NICE',
    28:'BAD',
    29:'GOOD'
}

while (cap.isOpened()):
    ret, img = cap.read()
    img = cv2.resize(img, (960, 540))
    x, y, c = img.shape
    t_elapsed = abs(sTime - time.time())

    if t_elapsed > 60:
        break

    cTime = time.time()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks != 0:
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                for id, lms in enumerate(handlms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lms.x * w), int(lms.y * h)
                    if id == 0:
                        pivot_x = int(lms.x * x)
                        pivot_y = int(lms.y * y)
                        mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
                        row_list.append(pivot_x)
                        row_list.append(pivot_y)
                    else:
                        lmx = int(lms.x * x)
                        lmy = int(lms.y * y)
                        mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
                        row_list.append(pivot_x - lmx)
                        row_list.append(pivot_y - lmy)

                break

            prediction = model.predict([row_list])
            probability = model.predict_proba([row_list])

            # Get the highest probability for the predicted gesture
            max_probability = max(probability[0]) * 100

            # Map numerical prediction to gesture label
            gesture_label = gesture_labels.get(prediction[0], 'Unknown')

            cv2.putText(img, 'Predicted gesture: {} (Probability: {:.2f}%)'.format(gesture_label, max_probability), org, font, fontScale=1, thickness=2,
                        color=color)

        row_list = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ret == True:
        cv2.imshow("result", img)

cap.release()
cv2.destroyAllWindows()
