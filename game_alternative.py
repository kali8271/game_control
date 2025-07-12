import cv2
import pygame
import sys
import numpy as np

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 400
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture-Controlled Game (OpenCV Version)")

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
clock = pygame.time.Clock()

char_x, char_y = 100, 300
char_width, char_height = 50, 50
jumping = False
speed = 5

# OpenCV setup
cap = cv2.VideoCapture(0)

def detect_hand_movement(frame):
    """Simple hand movement detection using contour analysis"""
    # Convert to HSV color space for better skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out very small contours
        if cv2.contourArea(largest_contour) > 5000:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate center of hand
            center_x = x + w // 2
            
            # Draw rectangle around hand
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Determine movement based on hand position
            frame_center = frame.shape[1] // 2
            
            if center_x < frame_center - 50:
                return "left"
            elif center_x > frame_center + 50:
                return "right"
            elif w * h > 20000:  # Large hand area indicates "jump" gesture
                return "jump"
            else:
                return "neutral"
    
    return "neutral"

def detect_fingers(frame):
    """Finger counting using convexity defects for better accuracy"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    # Find the largest contour (assumed to be the hand)
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 5000:
        return 0

    # Convex hull and defects
    hull = cv2.convexHull(max_contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(max_contour, hull)
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        # Use cosine rule to filter defects
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-5))
        if angle <= np.pi / 2 and d > 10000:  # angle less than 90 degree and reasonable depth
            finger_count += 1
            cv2.circle(frame, far, 8, [255,0,0], -1)

    # Draw the contour and hull for visualization
    cv2.drawContours(frame, [max_contour], -1, (0,255,0), 2)
    hull_points = cv2.convexHull(max_contour)
    cv2.drawContours(frame, [hull_points], -1, (0,0,255), 2)

    return min(finger_count + 1, 5)  # Add 1 for the thumb, cap at 5

print("Starting gesture-controlled game...")
print("Controls:")
print("- Move hand left/right to move character")
print("- Show 3+ fingers to jump")
print("- Press 'q' to quit")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    
    # Detect hand movement
    movement = detect_hand_movement(frame)
    
    # Detect finger count
    finger_count = detect_fingers(frame)
    
    # Game logic based on movement and finger count
    if movement == "right" or finger_count == 1:
        char_x += speed
    elif movement == "left" or finger_count == 2:
        char_x -= speed
    elif movement == "jump" or finger_count >= 3:
        jumping = True
    else:
        jumping = False

    # Keep character within bounds
    char_x = max(0, min(WIDTH - char_width, char_x))
    char_y = 250 if jumping else 300

    # Draw character
    win.fill(WHITE)
    pygame.draw.rect(win, GREEN, (char_x, char_y, char_width, char_height))
    
    # Draw instruction text
    font = pygame.font.Font(None, 36)
    text = font.render(f"Fingers: {finger_count} | Movement: {movement}", True, (0, 0, 0))
    win.blit(text, (10, 10))
    
    pygame.display.update()
    clock.tick(30)

    # Show webcam frame with detection info
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Movement: {movement}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture (OpenCV)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit() 