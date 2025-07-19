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
    script_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level
    header_folder = os.path.join(script_dir, "Header")
    
    if not os.path.exists(header_folder):
        print(f"Error: Header folder '{header_folder}' not found.")
        return
    
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
    draw_color = (0, 255, 0)  # Start with green for visibility

    header = cv2.resize(icons[0], (610, HEADER_HEIGHT))

    # For FPS display
    pTime = 0

    # Define color options
    colors = [
        (0, 255, 255),  # Yellow
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 0, 255),  # Purple
        (0, 0, 0)       # Black (Eraser)
    ]

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
            
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (610, 720))
        
        img = detector.findHands(img)
        lm_list, _ = detector.findPosition(img, draw=False)
        
        if lm_list:
            if len(lm_list) >= 13:
                x1, y1 = lm_list[8][1:]  # Index finger
                x2, y2 = lm_list[12][1:]  # Middle finger
                fingers_up = detector.fingersUp()
                

                # Selection mode
                if fingers_up[1] and fingers_up[2]:
                    if y1 < HEADER_HEIGHT:
                        icon_width = 610 // len(icons)
                        selected = x1 // icon_width
                        if selected < len(icons):
                            header = cv2.resize(icons[selected], (610, HEADER_HEIGHT))
                            draw_color = (0, 0, 0) if selected == len(icons) - 1 else (
                                colors[selected] if selected < len(colors) else draw_color
                            )
                    cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)
                    xp, yp = 0, 0  # Reset

                # Drawing mode
                elif fingers_up[1] and not fingers_up[2]:
                    print("Mode: Drawing")
                    thickness = ERASER_THICKNESS if draw_color == (0, 0, 0) else BRUSH_THICKNESS
                    cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                        continue  # Skip this frame

                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0  # Reset if fingers not in drawing mode

        # Simple merge using addWeighted for debug phase
        img = cv2.addWeighted(img, 0.5, canvas, 1, 0)

        # Add header
        img[0:HEADER_HEIGHT, 0:610] = header

        # FPS display
        cTime = time.time()
        fps = 1 / (cTime - pTime + 1e-5)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Virtual Painter", img)


        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            canvas = np.zeros((720, 610, 3), np.uint8)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
