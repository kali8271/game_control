import cv2
import mediapipe as mp
import pygame
import sys

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 400
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture-Controlled Game")

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
clock = pygame.time.Clock()

char_x, char_y = 100, 300
char_width, char_height = 50, 50
jumping = False
speed = 5

# OpenCV setup
cap = cv2.VideoCapture(0)
finger_tips = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks):
    count = 0
    landmarks = hand_landmarks.landmark
    if landmarks[4].x < landmarks[3].x:
        count += 1
    for tip in finger_tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        finger_count = count_fingers(hand_landmarks)

    # Game logic
    if finger_count == 1:
        char_x += speed
    elif finger_count == 2:
        char_x -= speed
    elif finger_count == 3:
        jumping = True
    else:
        jumping = False

    char_y = 250 if jumping else 300

    # Draw character
    win.fill(WHITE)
    pygame.draw.rect(win, GREEN, (char_x, char_y, char_width, char_height))
    pygame.display.update()
    clock.tick(30)

    # Show webcam frame
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
