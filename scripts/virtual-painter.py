import numpy as np
import cv2 
import os
import time
import HandTrackingModule as htm



brush_thickness = 25
eraser_thickness = 100

# load heade icnos
folder_name = r"E:\cv\NTI\vp\Virtual-Painter\Header"
icons_names = os.listdir(folder_name)
icons = []
for icon_path in icons_names:
    icon = cv2.imread(os.path.join(folder_name, icon_path))
    icons.append(icon)
header = icons[0]
header = cv2.resize(header, (610, header.shape[0])) # Resize header to match the width of the main image (610)
draw_color = (0, 0, 0)

# read video
cap = cv2.VideoCapture(0)
cap.set(3, 610) # width
cap.set(4, 720) # height

# detect hand
detector = htm.handDetector(detectionCon=0.8,maxHands=1)
canvas = np.zeros((720, 610, 3), np.uint8)

xp, yp = 0, 0

while True:
     ret, img = cap.read()
     img = cv2.flip(img, 1)

     # Resize img to ensure it matches the canvas size
     img = cv2.resize(img, (610, 720))

     # find hand landmarks
     img = detector.findHands(img)
     lm_list = detector.findPosition(img, draw=False)

     # Ensure enough landmarks are detected
     if len(lm_list) >= 13:
          # find tip of index and middle fingers
          x1, y1 = lm_list[8][1:]
          x2, y2 = lm_list[12][1:]

          # check which fingers are up
          fingers_up = detector.fingersUp()

          # selection mode -> 2 fingures up
          if fingers_up[1] and fingers_up[2]:
               if y1 < 120:
                    if 80 < x1 < 150:
                         header = icons[0]
                         draw_color = (0, 255, 255)
                    elif 170 < x1 < 240:
                         header = icons[1]
                         draw_color = (0, 255, 0)
                    elif 255 < x1 < 330:
                         header = icons[2]
                         draw_color = (255, 0, 0)
                    elif 350 < x1 < 420:
                         header = icons[3]
                         draw_color = (0, 0, 255)
                    elif 435 < x1 < 505:
                         header = icons[4]
                         draw_color = (255, 0, 255)
                    elif 530 < x1 < 600:
                         header = icons[5]
                         draw_color = (0, 0, 0)
               cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
                    
          # if drawing mode -> index fingure up
          if fingers_up[1] and fingers_up[2] == False:
               cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
               if xp == 0 and yp == 0:
                    xp, yp = x1, y1

               cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
               xp, yp = x1, y1

     gray_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
     _, inv = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
     inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)    
     img = cv2.bitwise_and(img, inv)
     img = cv2.bitwise_or(img, canvas)

     # Draw header at the top (now header width matches img width)
     img[0:header.shape[0], 0:header.shape[1]] = header

     cv2.imshow("hand", img)
     if cv2.waitKey(1) & 0xFF == 27:
               break
