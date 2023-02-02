import cv2
import mediapipe as mp
import time

# create video object
cap = cv2.VideoCapture(0)

# create an object from the class Hands()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0   # previous time
cTime = 0   # current time

while True:
    success, img = cap.read()
    # send rgb image to the object
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # open the results and extract imformation, e.g. multiple hands
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # get information of each hand
            for id, lm in enumerate(handLms.landmark):
                # the landmarks are the different points of the hands
                #print(id, lm)
                h, w, c = img.shape
                # find the center postion
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            # extract information of each hand
            # use mpDraw to draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
