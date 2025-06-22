#step num 1 import libraries

import cv2  # OpenCV for image processing
import numpy as np # NumPy for numerical operations
import mediapipe as mp  #detect hands

#step num 2 initialize mediapipe hands
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL  # For controlling audio volume
import math  # For mathematical operations

#step num 3 initialize mediapipe and control hands 

mpHands = mp.solution.hands
hands = mpHands.hands()
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume =interface.QueryInterface(IAudioEndpointVolume)

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

#step num 4 initialize webcam
cap = cv2.VideoCapture(0)

#step num 5 main loop to detect hands and control volume
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

#step num 6 landmark draw hands and get finger locationcontrol volume

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # h, w, c = img.shape
            # x1 = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * w)
            # y1 = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h)
            # x2 = int(handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x * w)
            # y2 = int(handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y * h)

#step num 7 calculate distance between thumb and index finger
x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
x2, y2 = lmList[4][1], lmList[4][2]  # Thumb tip
length = math.hypot(x2 - x1, y2 - y1)

#step num 8 map the length to volume range
vol = np.interp(length, [30, 200], [minVol, maxVol])
volume.SetMasterVolumeLevel(vol, None)

#step num 9 display the volume level on the screen
volBar = np.interp(length, [30, 200], [400, 150])
volPer = np.interp(length, [30, 200], [0, 100])
cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
cv2.putText(img, f'Volume: {int(volPer)}%', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #step num 10 show the image landmarks and volume control
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#step num 11 release resources
cap.release()
cv2.destroyAllWindows() 