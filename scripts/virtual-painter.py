import numpy as np
import cv2 
import os
import time
import HandTrackingModule as htm

# Constants
BRUSH_THICKNESS = 25
ERASER_THICKNESS = 100
HEADER_HEIGHT = 120

def load_icons(folder_path):
    icons = []
    for icon_name in sorted(os.listdir(folder_path)):
        icon_path = os.path.join(folder_path, icon_name)
        icon = cv2.imread(icon_path)
        if icon is not None:
            icons.append(icon)
    return icons

def main():
    # Load header icons
    script_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from scripts folder
    header_folder = os.path.join(script_dir, "Header")
    icons = load_icons(header_folder)
    if not icons:
        print("Error: No icons loaded")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(3, 610)  # width
    cap.set(4, 720)  # height
    
    detector = htm.handDetector(detectionCon=0.8, maxHands=1)
    canvas = np.zeros((720, 610, 3), np.uint8)
    xp, yp = 0, 0
    draw_color = (0, 0, 0)  # Start with black
    
    # Resize header to match camera width
    header = cv2.resize(icons[0], (610, HEADER_HEIGHT))
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
            
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (610, 720))
        
        # Find hand landmarks
        img = detector.findHands(img)
        lm_list = detector.findPosition(img, draw=False)
        
        if len(lm_list) >= 13:
            x1, y1 = lm_list[8][1:]  # Index finger
            x2, y2 = lm_list[12][1:]  # Middle finger
            fingers_up = detector.fingersUp()
            
            # Selection mode - two fingers up
            if fingers_up[1] and fingers_up[2]:
                if y1 < HEADER_HEIGHT:
                    # Check which icon is selected
                    icon_width = 610 // len(icons)
                    selected = x1 // icon_width
                    if selected < len(icons):
                        header = cv2.resize(icons[selected], (610, HEADER_HEIGHT))
                        if selected == len(icons) - 1:  # Last icon is eraser
                            draw_color = (0, 0, 0)
                        else:
                            # Assign colors based on position
                            colors = [
                                (0, 255, 255),  # Yellow
                                (0, 255, 0),    # Green
                                (255, 0, 0),    # Blue
                                (0, 0, 255),    # Red
                                (255, 0, 255),  # Purple
                                (0, 0, 0)       # Black (eraser)
                            ]
                            draw_color = colors[selected]
                
                cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)
                xp, yp = 0, 0  # Reset drawing position
                
            # Drawing mode - index finger up
            elif fingers_up[1] and not fingers_up[2]:
                thickness = ERASER_THICKNESS if draw_color == (0, 0, 0) else BRUSH_THICKNESS
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                # Draw on canvas
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0  # Reset when no fingers are up
        
        # Merge canvas with camera feed
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, inv)
        img = cv2.bitwise_or(img, canvas)
        
        # Add header
        img[0:HEADER_HEIGHT, 0:610] = header
        
        cv2.imshow("Virtual Painter", img)
        

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('c'):  # Clear canvas
            canvas = np.zeros((720, 610, 3), np.uint8)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()