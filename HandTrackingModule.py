import cv2
import mediapipe as mp
import time


class handDetector():

    def __init__(self,
                 mode=False,
                 num_hands=2,
                 complexity=1,
                 min_detection=0.5,
                 min_tracking=0.5):
        self.mode = mode
        self.num_hands = num_hands
        self.complexity = 1
        self.min_detection = min_detection
        self.min_tracking = min_tracking

        # create an object from the class Hands()
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.num_hands, self.complexity, self.min_detection,
                                        self.min_tracking)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # send rgb image to the object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # open the results and extract imformation, e.g. multiple hands
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # extract information of each hand
                # use mpDraw to draw landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # get information of each hand
            for idx, lm in enumerate(myHand.landmark):
                # the landmarks are the different points of the hands
                # print(id, lm)
                h, w, c = img.shape
                # find the center position
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    # create video object
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            # print the position of thumb
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0))

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()